use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "embeddy")]
#[command(version, about = "A lightweight embeddings-only model runtime", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Download a model from HuggingFace
    Pull {
        /// HuggingFace model repository ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        model: String,

        /// Optional alias for the model
        #[arg(long)]
        alias: Option<String>,
    },

    /// Start the HTTP API server (models loaded on-demand)
    Serve {
        /// Device to run on (e.g., "cpu" or "cuda:0")
        #[arg(long, default_value = "cpu", env = "EMBEDDY_DEVICE")]
        device: String,

        /// Port to listen on
        #[arg(long, default_value = "8080", env = "EMBEDDY_PORT")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0", env = "EMBEDDY_HOST")]
        host: String,

        /// Model to preload at startup
        #[arg(long, env = "EMBEDDY_MODEL")]
        model: Option<String>,

        /// Embedding cache size (number of entries)
        #[arg(long, default_value = "10000", env = "EMBEDDY_CACHE_SIZE")]
        cache_size: usize,
    },

    /// Run embeddings on text input
    Run {
        /// Model name or alias to use
        model: String,

        /// Text to embed (can be specified multiple times)
        #[arg(long)]
        text: Vec<String>,

        /// Device to run on (e.g., "cpu" or "cuda:0")
        #[arg(long, default_value = "cpu")]
        device: String,
    },

    /// List installed models
    List,

    /// Remove an installed model
    Remove {
        /// Model name or alias to remove
        model: String,
    },
}
