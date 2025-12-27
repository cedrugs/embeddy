mod cli;
mod config;
mod embedder;
mod error;
mod model;
mod server;

use candle_core::Device;
use clap::Parser;
use cli::{Cli, Commands};
use config::Config;
use error::Result;
use model::ModelDownloader;

fn parse_device(device_str: &str) -> Result<Device> {
    match device_str {
        "cpu" => Ok(Device::Cpu),
        s if s.starts_with("cuda") => {
            let parts: Vec<&str> = s.split(':').collect();
            let ordinal = if parts.len() > 1 {
                parts[1].parse::<usize>().map_err(|_| {
                    error::Error::InvalidInput(format!("Invalid CUDA device: {}", s))
                })?
            } else {
                0
            };
            Device::new_cuda(ordinal).map_err(|e| {
                error::Error::Config(format!("Failed to initialize CUDA device: {}", e))
            })
        }
        _ => Err(error::Error::InvalidInput(format!(
            "Unknown device: {}",
            device_str
        ))),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let config = Config::from_env()?;

    match cli.command {
        Commands::Pull { model, alias } => {
            let mut downloader = ModelDownloader::new(config)?;
            let model_info = downloader.pull(&model, alias)?;

            println!("✓ Successfully pulled model: {}", model);
            println!("  Repository: {}", model_info.hf_repo_id);
            println!("  Path: {:?}", model_info.model_path);
            if let Some(alias) = model_info.alias {
                println!("  Alias: {}", alias);
            }
        }

        Commands::Serve { device, port, host, model, cache_size } => {
            let device = parse_device(&device)?;
            let device_name = format!("{:?}", device);

            let state = server::AppState::new(config.clone(), device, cache_size);

            // Preload model if specified
            if let Some(ref model_name) = model {
                // Check if model exists, if not try to download it
                let registry = model::ModelRegistry::load(&config)?;
                if registry.get_model(model_name).is_err() {
                    println!("📥 Model '{}' not found, downloading...", model_name);
                    let mut downloader = ModelDownloader::new(config.clone())?;
                    downloader.pull(model_name, None)?;
                    println!("✓ Download complete");
                }

                println!("📦 Preloading model: {}", model_name);
                state.get_or_load_embedder(model_name).await?;
                println!("✓ Model loaded");
            }

            println!("🚀 Embeddy server starting...");
            println!("   Device: {}", device_name);
            println!("   Cache size: {}", cache_size);
            println!("   Listening on: http://{}:{}", host, port);
            println!("   Docs: http://{}:{}/docs", host, port);
            if model.is_none() {
                println!("\n   Models will be loaded on-demand when requested via API");
            }

            server::serve(&host, port, state).await?;
        }

        Commands::Run {
            model,
            text,
            device,
        } => {
            if text.is_empty() {
                return Err(error::Error::InvalidInput(
                    "No text provided. Use --text \"your text\"".to_string(),
                ));
            }

            let registry = model::ModelRegistry::load(&config)?;
            let model_info = registry.get_model(&model)?;

            let device = parse_device(&device)?;

            tracing::info!("Loading model '{}'", model);
            let embedder = embedder::Embedder::load(model_info, device)?;

            tracing::info!("Generating embeddings for {} texts", text.len());
            let embeddings = embedder.embed(&text)?;

            let output = serde_json::json!({
                "model": model,
                "dimension": embedder.embedding_dim(),
                "embeddings": embeddings,
            });

            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }

        Commands::List => {
            let registry = model::ModelRegistry::load(&config)?;
            let models = registry.list_models();

            if models.is_empty() {
                println!("No models installed.");
                println!("Use 'embeddy pull <model-id>' to download a model.");
            } else {
                println!("Installed models:\n");
                for model in models {
                    println!("  {}", model.alias.as_ref().unwrap_or(&model.name));
                    println!("    Repository: {}", model.hf_repo_id);
                    println!("    Path: {:?}", model.model_path);
                    println!("    Downloaded: {}", model.downloaded_at);
                    if let Some(dim) = model.embedding_dim {
                        println!("    Dimension: {}", dim);
                    }
                    println!();
                }
            }
        }

        Commands::Remove { model } => {
            let mut registry = model::ModelRegistry::load(&config)?;
            let model_info = registry.remove_model(&model)?;

            // Remove model files
            if model_info.model_path.exists() {
                std::fs::remove_dir_all(&model_info.model_path).ok();
            }

            registry.save(&config)?;
            println!("✓ Removed model: {}", model);
        }
    }

    Ok(())
}
