use crate::config::Config;
use crate::error::{Error, Result};
use crate::model::{ModelInfo, ModelRegistry};
use candle_core::pickle;
use hf_hub::api::sync::Api;
use std::path::Path;

pub struct ModelDownloader {
    config: Config,
    registry: ModelRegistry,
}

impl ModelDownloader {
    pub fn new(config: Config) -> Result<Self> {
        let registry = ModelRegistry::load(&config)?;
        Ok(Self { config, registry })
    }

    pub fn pull(&mut self, hf_repo_id: &str, alias: Option<String>) -> Result<ModelInfo> {
        tracing::info!("Pulling model from HuggingFace: {}", hf_repo_id);

        let api = Api::new().map_err(|e| Error::DownloadFailed(e.to_string()))?;

        let repo = api.model(hf_repo_id.to_string());

        tracing::info!("Downloading model files...");

        let model_file = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .map_err(|e| Error::DownloadFailed(format!("Could not find model file: {}", e)))?;

        let _tokenizer_file = repo
            .get("tokenizer.json")
            .map_err(|e| Error::DownloadFailed(format!("Could not find tokenizer: {}", e)))?;

        let _config_file = repo
            .get("config.json")
            .map_err(|e| Error::DownloadFailed(format!("Could not find config: {}", e)))?;

        let model_dir = model_file
            .parent()
            .ok_or_else(|| Error::DownloadFailed("Invalid model path".to_string()))?;

        // Auto-convert PyTorch to SafeTensors if needed
        Self::ensure_safetensors(model_dir)?;

        let name = alias.clone().unwrap_or_else(|| {
            hf_repo_id
                .split('/')
                .next_back()
                .unwrap_or(hf_repo_id)
                .to_string()
        });

        let model_info = ModelInfo {
            name: hf_repo_id.to_string(),
            hf_repo_id: hf_repo_id.to_string(),
            alias,
            model_path: model_dir.to_path_buf(),
            embedding_dim: None,
            downloaded_at: chrono::Utc::now().to_rfc3339(),
        };

        self.registry.add_model(model_info.clone());
        self.registry.save(&self.config)?;

        tracing::info!("Model '{}' successfully pulled and registered", name);

        Ok(model_info)
    }

    fn ensure_safetensors(model_dir: &Path) -> Result<()> {
        let pytorch_file = model_dir.join("pytorch_model.bin");
        let safetensors_file = model_dir.join("model.safetensors");

        // If safetensors exists, we're good
        if safetensors_file.exists() {
            return Ok(());
        }

        // If pytorch file doesn't exist, nothing to convert
        if !pytorch_file.exists() {
            return Ok(());
        }

        tracing::info!("Converting pytorch_model.bin to model.safetensors...");

        // Read PyTorch file and load all tensors
        let tensors_vec = pickle::read_all(&pytorch_file)
            .map_err(|e| Error::ModelLoadFailed(format!("Failed to read PyTorch file: {}", e)))?;

        tracing::info!("Loading {} tensors from PyTorch model", tensors_vec.len());

        // Convert to HashMap
        let tensors: std::collections::HashMap<_, _> = tensors_vec.into_iter().collect();

        // Save as safetensors
        candle_core::safetensors::save(&tensors, &safetensors_file)
            .map_err(|e| Error::ModelLoadFailed(format!("Failed to save SafeTensors: {}", e)))?;

        tracing::info!("✓ Converted to SafeTensors format");

        // Remove the old PyTorch file to save space
        if let Err(e) = std::fs::remove_file(&pytorch_file) {
            tracing::warn!("Could not remove pytorch_model.bin: {}", e);
        } else {
            tracing::info!("Removed pytorch_model.bin to save space");
        }

        Ok(())
    }
}
