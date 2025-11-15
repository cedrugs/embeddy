use crate::config::Config;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub hf_repo_id: String,
    pub alias: Option<String>,
    pub model_path: PathBuf,
    pub embedding_dim: Option<usize>,
    pub downloaded_at: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    pub fn load(config: &Config) -> Result<Self> {
        if !config.registry_path.exists() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(&config.registry_path)?;
        let registry: ModelRegistry = toml::from_str(&content)?;
        Ok(registry)
    }

    pub fn save(&self, config: &Config) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        fs::write(&config.registry_path, content)?;
        Ok(())
    }

    pub fn add_model(&mut self, model: ModelInfo) {
        let key = model.alias.clone().unwrap_or_else(|| model.name.clone());
        self.models.insert(key, model);
    }

    pub fn get_model(&self, name: &str) -> Result<&ModelInfo> {
        self.models
            .get(name)
            .ok_or_else(|| Error::ModelNotFound(name.to_string()))
    }

    pub fn list_models(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }
}
