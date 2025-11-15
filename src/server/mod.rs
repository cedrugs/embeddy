use crate::embedder::Embedder;
use crate::error::{Error, Result};
use axum::{
	extract::State,
	http::StatusCode,
	response::{IntoResponse, Response},
	routing::{get, post},
	Json, Router,
};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
	embedder: Arc<RwLock<Embedder>>,
	model_name: String,
	device_name: String,
}

impl AppState {
	pub fn new(embedder: Embedder, model_name: String, device_name: String) -> Self {
		Self {
			embedder: Arc::new(RwLock::new(embedder)),
			model_name,
			device_name,
		}
	}
}

#[derive(Serialize)]
pub struct HealthResponse {
	pub status: String,
	pub model: String,
	pub device: String,
	pub embedding_dim: usize,
}

#[derive(Deserialize)]
pub struct EmbedRequest {
	#[serde(default)]
	pub model: Option<String>,
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
			_ => (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error".to_string()),
		};

		let body = Json(serde_json::json!({
			"error": message,
		}));

		(status, body).into_response()
	}
}

async fn health_handler(State(state): State<AppState>) -> Result<Json<HealthResponse>> {
	let embedder = state.embedder.read().await;
	
	Ok(Json(HealthResponse {
		status: "ok".to_string(),
		model: state.model_name.clone(),
		device: state.device_name.clone(),
		embedding_dim: embedder.embedding_dim(),
	}))
}

async fn embed_handler(
	State(state): State<AppState>,
	Json(payload): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>> {
	if payload.input.is_empty() {
		return Err(Error::InvalidInput("Input cannot be empty".to_string()));
	}

	let embedder = state.embedder.read().await;
	let embeddings = embedder.embed(&payload.input)?;
	let dimension = embedder.embedding_dim();

	Ok(Json(EmbedResponse {
		model: state.model_name.clone(),
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
