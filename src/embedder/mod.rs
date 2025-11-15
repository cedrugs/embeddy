use crate::error::{Error, Result};
use crate::model::ModelInfo;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;

pub struct Embedder {
    model_type: String,
    model_path: PathBuf,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: Device,
    embedding_dim: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
}

impl Embedder {
    pub fn load(model_info: &ModelInfo, device: Device) -> Result<Self> {
        tracing::info!("Loading model from: {:?}", model_info.model_path);

        let config_path = model_info.model_path.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::ModelLoadFailed(format!("Failed to read config: {}", e)))?;

        let config: Value = serde_json::from_str(&config_content)
            .map_err(|e| Error::ModelLoadFailed(format!("Failed to parse config: {}", e)))?;

        let model_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("bert")
            .to_string();

        let embedding_dim = config
            .get("hidden_size")
            .or_else(|| config.get("n_embd"))
            .or_else(|| config.get("dim"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                Error::ModelLoadFailed("Could not determine embedding dimension".to_string())
            })? as usize;

        let num_hidden_layers = config
            .get("num_hidden_layers")
            .or_else(|| config.get("n_layer"))
            .and_then(|v| v.as_u64())
            .unwrap_or(12) as usize;

        let num_attention_heads = config
            .get("num_attention_heads")
            .or_else(|| config.get("n_head"))
            .and_then(|v| v.as_u64())
            .unwrap_or(12) as usize;

        let model_file = model_info.model_path.join("model.safetensors");
        let model_file = if model_file.exists() {
            model_file
        } else {
            model_info.model_path.join("pytorch_model.bin")
        };

        let tokenizer_path = model_info.model_path.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::ModelLoadFailed(format!("Failed to load tokenizer: {}", e)))?;

        tracing::info!("Model loaded successfully");
        tracing::info!("  Type: {}", model_type);
        tracing::info!("  Embedding dimension: {}", embedding_dim);
        tracing::info!("  Hidden layers: {}", num_hidden_layers);
        tracing::info!("  Attention heads: {}", num_attention_heads);

        Ok(Self {
            model_type,
            model_path: model_file,
            tokenizer: Arc::new(tokenizer),
            device,
            embedding_dim,
            num_hidden_layers,
            num_attention_heads,
        })
    }

    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Err(Error::InvalidInput("Empty input texts".to_string()));
        }

        tracing::debug!("Encoding {} texts", texts.len());

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| Error::EmbeddingError(format!("Tokenization failed: {}", e)))?;

            let token_ids = encoding.get_ids();

            let embeddings = self.embed_tokens(token_ids)?;

            let pooled = embeddings
                .mean(0)
                .map_err(|e| Error::EmbeddingError(format!("Pooling failed: {}", e)))?;

            let embedding_vec = pooled
                .to_vec1::<f32>()
                .map_err(|e| Error::EmbeddingError(format!("Failed to convert to vec: {}", e)))?;

            all_embeddings.push(embedding_vec);
        }

        Ok(all_embeddings)
    }

    fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor> {
        let safetensors = unsafe {
            candle_core::safetensors::MmapedSafetensors::multi(&[self.model_path.clone()])
                .map_err(|e| Error::ModelLoadFailed(format!("Failed to load safetensors: {}", e)))?
        };

        // Get list of tensors and find embedding weight
        let tensor_list = safetensors.tensors();
        let embedding_weight_name = tensor_list
            .iter()
            .find(|(name, _)| {
                name.contains("embeddings")
                    && name.contains("word_embeddings")
                    && name.ends_with("weight")
                    || name.ends_with("embed_tokens.weight")
                    || name.ends_with("wte.weight")
            })
            .map(|(name, _)| name.clone())
            .ok_or_else(|| {
                Error::ModelLoadFailed("Could not find embedding weight tensor".to_string())
            })?;

        tracing::debug!("Using embedding tensor: {}", embedding_weight_name);

        let embeddings_weight = safetensors
            .load(&embedding_weight_name, &self.device)
            .map_err(|e| {
                Error::EmbeddingError(format!("Failed to load embedding tensor: {}", e))
            })?;

        let token_ids_tensor = Tensor::new(token_ids, &self.device)
            .map_err(|e| Error::EmbeddingError(format!("Failed to create token tensor: {}", e)))?;

        let token_embeddings = embeddings_weight
            .index_select(&token_ids_tensor, 0)
            .map_err(|e| Error::EmbeddingError(format!("Failed to index embeddings: {}", e)))?;

        Ok(token_embeddings)
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
