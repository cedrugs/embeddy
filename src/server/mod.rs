use crate::config::Config;
use crate::embedder::Embedder;
use crate::error::{Error, Result};
use crate::model::ModelRegistry;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
    embedders: Arc<RwLock<HashMap<String, Embedder>>>,
    config: Config,
    device: Device,
}

impl AppState {
    pub fn new(config: Config, device: Device) -> Self {
        Self {
            embedders: Arc::new(RwLock::new(HashMap::new())),
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

        // Load the model
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
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Error::ModelNotFound(_) => (StatusCode::NOT_FOUND, self.to_string()),
            Error::InvalidInput(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            Error::ModelLoadFailed(_) | Error::EmbeddingError(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error".to_string(),
            ),
        };

        let body = Json(serde_json::json!({
            "error": message,
        }));

        (status, body).into_response()
    }
}

async fn health_handler(State(state): State<AppState>) -> Result<Json<HealthResponse>> {
    let embedders = state.embedders.read().await;
    let loaded_models: Vec<String> = embedders.keys().cloned().collect();

    Ok(Json(HealthResponse {
        status: "ok".to_string(),
        loaded_models,
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

    // Load model if not already loaded
    state.get_or_load_embedder(&payload.model).await?;

    let embedders = state.embedders.read().await;
    let embedder = embedders
        .get(&payload.model)
        .ok_or_else(|| Error::ModelNotFound(payload.model.clone()))?;

    let embeddings = embedder.embed(&payload.input)?;
    let dimension = embedder.embedding_dim();

    Ok(Json(EmbedResponse {
        model: payload.model,
        dimension,
        embeddings,
    }))
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health_handler))
        .route("/api/embed", post(embed_handler))
        .with_state(state)
}

pub async fn serve(host: &str, port: u16, state: AppState) -> Result<()> {
    let app = create_router(state);
    let addr = format!("{}:{}", host, port);

    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| Error::ConfigError(format!("Failed to bind to {}: {}", addr, e)))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| Error::ConfigError(format!("Server error: {}", e)))?;

    Ok(())
}
