use crate::config::Config;
use crate::error::{Error, Result};
use crate::model::{ModelInfo, ModelRegistry};
use hf_hub::api::sync::Api;

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
}
