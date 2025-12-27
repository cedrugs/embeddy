use crate::config::Config;
use crate::embedder::Embedder;
use crate::error::{Error, Result};
use crate::model::ModelRegistry;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::sync::RwLock;

// Simple LRU cache
struct LruCache<K, V> {
    capacity: usize,
    map: HashMap<K, (V, u64)>,
    counter: u64,
}

impl<K: Eq + Hash + Clone, V: Clone> LruCache<K, V> {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            counter: 0,
        }
    }

    fn get(&mut self, key: &K) -> Option<V> {
        if let Some((value, access)) = self.map.get_mut(key) {
            self.counter += 1;
            *access = self.counter;
            Some(value.clone())
        } else {
            None
        }
    }

    fn insert(&mut self, key: K, value: V) {
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            // Evict least recently used
            if let Some(lru_key) = self
                .map
                .iter()
                .min_by_key(|(_, (_, access))| access)
                .map(|(k, _)| k.clone())
            {
                self.map.remove(&lru_key);
            }
        }
        self.counter += 1;
        self.map.insert(key, (value, self.counter));
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct CacheKey {
    model: String,
    text_hash: u64,
}

fn hash_text(text: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

#[derive(Clone)]
pub struct AppState {
    embedders: Arc<RwLock<HashMap<String, Embedder>>>,
    cache: Arc<RwLock<LruCache<CacheKey, Vec<f32>>>>,
    config: Config,
    device: Device,
}

impl AppState {
    pub fn new(config: Config, device: Device, cache_size: usize) -> Self {
        Self {
            embedders: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
            config,
            device,
        }
    }

    pub async fn get_or_load_embedder(&self, model_name: &str) -> Result<()> {
        let embedders = self.embedders.read().await;
        if embedders.contains_key(model_name) {
            return Ok(());
        }
        drop(embedders);

        let registry = ModelRegistry::load(&self.config)?;
        let model_info = registry.get_model(model_name)?;

        tracing::info!(
            "Loading model '{}' on device '{:?}'",
            model_name,
            self.device
        );
        let embedder = Embedder::load(model_info, self.device.clone())?;

        let mut embedders = self.embedders.write().await;
        embedders.insert(model_name.to_string(), embedder);

        Ok(())
    }
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub loaded_models: Vec<String>,
    pub cache_entries: usize,
    pub device: String,
}

#[derive(Deserialize)]
pub struct EmbedRequest {
    pub model: String,
    pub input: Vec<String>,
}

#[derive(Serialize)]
pub struct EmbedResponse {
    pub model: String,
    pub dimension: usize,
    pub embeddings: Vec<Vec<f32>>,
    pub cache_hits: usize,
}

#[derive(Serialize)]
pub struct ModelListResponse {
    pub models: Vec<ModelEntry>,
}

#[derive(Serialize)]
pub struct ModelEntry {
    pub name: String,
    pub alias: Option<String>,
    pub hf_repo_id: String,
    pub embedding_dim: Option<usize>,
    pub downloaded_at: String,
}

#[derive(Serialize)]
pub struct MessageResponse {
    pub message: String,
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Error::ModelNotFound(_) => (StatusCode::NOT_FOUND, self.to_string()),
            Error::InvalidInput(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            Error::ModelLoadFailed(_) | Error::Embedding(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error".to_string(),
            ),
        };

        let body = Json(serde_json::json!({ "error": message }));
        (status, body).into_response()
    }
}

async fn health_handler(State(state): State<AppState>) -> Result<Json<HealthResponse>> {
    let embedders = state.embedders.read().await;
    let loaded_models: Vec<String> = embedders.keys().cloned().collect();
    let cache = state.cache.read().await;

    Ok(Json(HealthResponse {
        status: "ok".to_string(),
        loaded_models,
        cache_entries: cache.len(),
        device: format!("{:?}", state.device),
    }))
}

async fn embed_handler(
    State(state): State<AppState>,
    Json(payload): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>> {
    if payload.input.is_empty() {
        return Err(Error::InvalidInput("Input cannot be empty".to_string()));
    }

    state.get_or_load_embedder(&payload.model).await?;

    let mut embeddings = Vec::with_capacity(payload.input.len());
    let mut cache_hits = 0;
    let mut texts_to_embed: Vec<(usize, String)> = Vec::new();

    // Check cache for each input
    {
        let mut cache = state.cache.write().await;
        for (i, text) in payload.input.iter().enumerate() {
            let key = CacheKey {
                model: payload.model.clone(),
                text_hash: hash_text(text),
            };
            if let Some(cached) = cache.get(&key) {
                embeddings.push((i, cached));
                cache_hits += 1;
            } else {
                texts_to_embed.push((i, text.clone()));
            }
        }
    }

    // Embed cache misses
    if !texts_to_embed.is_empty() {
        let embedders = state.embedders.read().await;
        let embedder = embedders
            .get(&payload.model)
            .ok_or_else(|| Error::ModelNotFound(payload.model.clone()))?;

        let texts: Vec<String> = texts_to_embed.iter().map(|(_, t)| t.clone()).collect();
        let new_embeddings = embedder.embed(&texts)?;

        let mut cache = state.cache.write().await;
        for ((i, text), emb) in texts_to_embed.into_iter().zip(new_embeddings) {
            let key = CacheKey {
                model: payload.model.clone(),
                text_hash: hash_text(&text),
            };
            cache.insert(key, emb.clone());
            embeddings.push((i, emb));
        }
    }

    // Sort by original index
    embeddings.sort_by_key(|(i, _)| *i);
    let embeddings: Vec<Vec<f32>> = embeddings.into_iter().map(|(_, e)| e).collect();
    let dimension = embeddings.first().map(|e| e.len()).unwrap_or(0);

    Ok(Json(EmbedResponse {
        model: payload.model,
        dimension,
        embeddings,
        cache_hits,
    }))
}

async fn list_models_handler(State(state): State<AppState>) -> Result<Json<ModelListResponse>> {
    let registry = ModelRegistry::load(&state.config)?;
    let models = registry
        .list_models()
        .into_iter()
        .map(|m| ModelEntry {
            name: m.name.clone(),
            alias: m.alias.clone(),
            hf_repo_id: m.hf_repo_id.clone(),
            embedding_dim: m.embedding_dim,
            downloaded_at: m.downloaded_at.clone(),
        })
        .collect();

    Ok(Json(ModelListResponse { models }))
}

async fn delete_model_handler(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<MessageResponse>> {
    {
        let mut embedders = state.embedders.write().await;
        embedders.remove(&name);
    }

    let mut registry = ModelRegistry::load(&state.config)?;
    let model_info = registry.remove_model(&name)?;

    if model_info.model_path.exists() {
        std::fs::remove_dir_all(&model_info.model_path).ok();
    }

    registry.save(&state.config)?;

    Ok(Json(MessageResponse {
        message: format!("Model '{}' removed", name),
    }))
}

async fn openapi_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "openapi": "3.0.3",
        "info": {
            "title": "Embeddy API",
            "description": "A lightweight embeddings-only model runtime",
            "version": "0.1.0"
        },
        "paths": {
            "/api/health": {
                "get": {
                    "summary": "Health check",
                    "description": "Returns server status, loaded models, and cache stats",
                    "responses": {
                        "200": {
                            "description": "Server is healthy",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/HealthResponse" }
                                }
                            }
                        }
                    }
                }
            },
            "/api/embed": {
                "post": {
                    "summary": "Generate embeddings",
                    "description": "Generate embeddings for input texts. Results are cached.",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/EmbedRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Embeddings generated",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/EmbedResponse" }
                                }
                            }
                        },
                        "400": { "description": "Invalid input" },
                        "404": { "description": "Model not found" }
                    }
                }
            },
            "/api/models": {
                "get": {
                    "summary": "List models",
                    "description": "List all installed models",
                    "responses": {
                        "200": {
                            "description": "List of models",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/ModelListResponse" }
                                }
                            }
                        }
                    }
                }
            },
            "/api/models/{name}": {
                "delete": {
                    "summary": "Remove model",
                    "description": "Remove an installed model from registry and disk",
                    "parameters": [{
                        "name": "name",
                        "in": "path",
                        "required": true,
                        "schema": { "type": "string" }
                    }],
                    "responses": {
                        "200": { "description": "Model removed" },
                        "404": { "description": "Model not found" }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": { "type": "string" },
                        "loaded_models": { "type": "array", "items": { "type": "string" } },
                        "cache_entries": { "type": "integer" },
                        "device": { "type": "string" }
                    }
                },
                "EmbedRequest": {
                    "type": "object",
                    "required": ["model", "input"],
                    "properties": {
                        "model": { "type": "string", "description": "Model name or alias" },
                        "input": { "type": "array", "items": { "type": "string" }, "description": "Texts to embed" }
                    }
                },
                "EmbedResponse": {
                    "type": "object",
                    "properties": {
                        "model": { "type": "string" },
                        "dimension": { "type": "integer" },
                        "embeddings": { "type": "array", "items": { "type": "array", "items": { "type": "number" } } },
                        "cache_hits": { "type": "integer" }
                    }
                },
                "ModelListResponse": {
                    "type": "object",
                    "properties": {
                        "models": { "type": "array", "items": { "$ref": "#/components/schemas/ModelEntry" } }
                    }
                },
                "ModelEntry": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "alias": { "type": "string", "nullable": true },
                        "hf_repo_id": { "type": "string" },
                        "embedding_dim": { "type": "integer", "nullable": true },
                        "downloaded_at": { "type": "string" }
                    }
                }
            }
        }
    }))
}

async fn docs_handler() -> Html<&'static str> {
    Html(
        r##"<!DOCTYPE html>
<html>
<head>
    <title>Embeddy API</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
        SwaggerUIBundle({ url: "/openapi.json", dom_id: "#swagger-ui" });
    </script>
</body>
</html>"##,
    )
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health_handler))
        .route("/api/embed", post(embed_handler))
        .route("/api/models", get(list_models_handler))
        .route("/api/models/{name}", delete(delete_model_handler))
        .route("/openapi.json", get(openapi_handler))
        .route("/docs", get(docs_handler))
        .with_state(state)
}

pub async fn serve(host: &str, port: u16, state: AppState) -> Result<()> {
    let app = create_router(state);
    let addr = format!("{}:{}", host, port);

    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| Error::Config(format!("Failed to bind to {}: {}", addr, e)))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| Error::Config(format!("Server error: {}", e)))?;

    Ok(())
}
