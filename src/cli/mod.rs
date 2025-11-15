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
		#[arg(long, default_value = "cpu")]
		device: String,

		/// Port to listen on
		#[arg(long, default_value = "8080")]
		port: u16,

		/// Host to bind to
		#[arg(long, default_value = "0.0.0.0")]
		host: String,
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
}
