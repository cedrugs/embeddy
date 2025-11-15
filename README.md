# embeddy

[![Rust](https://img.shields.io/badge/rust-1.91%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://img.shields.io/github/actions/workflow/status/cedrugs/embeddy/ci.yml?branch=main)](https://github.com/cedrugs/embeddy/actions)

A lightweight, embeddings-only model runtime with CLI and HTTP API. Built in Rust for performance and efficiency, embeddy allows you to download, manage, and run text embedding models from HuggingFace without heavy dependencies.

## Features

- **Lightweight Runtime**: Pure Rust implementation using the Candle ML framework
- **HuggingFace Integration**: Download and cache models directly from HuggingFace Hub
- **Dynamic Model Loading**: Load multiple models on-demand via API without restart
- **Flexible Deployment**: Run as CLI tool or HTTP API server
- **Hardware Support**: CPU and CUDA GPU acceleration
- **Model Management**: Built-in registry for tracking and aliasing models
- **Docker Ready**: Includes Dockerfile and docker-compose configuration
- **Multiple Formats**: Supports both SafeTensors and PyTorch model formats

## Installation

### From Source

Requires Rust 1.91 or higher.

```bash
# Clone the repository
git clone https://github.com/cedrugs/embeddy.git
cd embeddy

# Build the project
cargo build --release

# The binary will be available at target/release/embeddy
```

### Using Docker

```bash
# Build the Docker image
docker build -t embeddy:latest .

# Or use docker-compose
docker-compose up -d
```

## Quick Start

### Docker (Recommended)

```bash
# Start the server (models loaded on-demand)
docker-compose up -d

# Download a model
docker exec embeddy embeddy pull sentence-transformers/all-MiniLM-L6-v2 --alias minilm

# Test the API
curl -X POST http://localhost:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minilm",
    "input": ["Hello, world!", "How are you?"]
  }'
```

### Local Installation

1. **Download a model from HuggingFace**:

```bash
embeddy pull sentence-transformers/all-MiniLM-L6-v2 --alias minilm
```

2. **Generate embeddings via CLI**:

```bash
embeddy run minilm --text "Hello, world!" --text "How are you?"
```

3. **Start the HTTP API server**:

```bash
embeddy serve --port 8080
```

4. **Make API requests** (models loaded on-demand):

```bash
curl -X POST http://localhost:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minilm",
    "input": ["Hello, world!", "How are you?"]
  }'
```

## Usage

### CLI Commands

#### Pull a Model

Download a model from HuggingFace Hub:

```bash
embeddy pull <MODEL_REPO_ID> [--alias <ALIAS>]
```

Examples:

```bash
# Download with repository ID
embeddy pull sentence-transformers/all-MiniLM-L6-v2

# Download with custom alias
embeddy pull sentence-transformers/all-mpnet-base-v2 --alias mpnet
```

#### List Installed Models

View all downloaded models:

```bash
embeddy list
```

#### Run Embeddings (CLI)

Generate embeddings for text inputs:

```bash
embeddy run <MODEL_NAME> --text <TEXT> [--text <TEXT>...] [--device <DEVICE>]
```

Options:
- `--text`: Text to embed (can be specified multiple times)
- `--device`: Device to run on (default: `cpu`, options: `cpu`, `cuda:0`, `cuda:1`, etc.)

Examples:

```bash
# Single text
embeddy run minilm --text "Machine learning is fascinating"

# Multiple texts
embeddy run minilm \
  --text "First sentence" \
  --text "Second sentence" \
  --text "Third sentence"

# Use GPU
embeddy run minilm --text "Hello" --device cuda:0
```

#### Serve HTTP API

Start the embedding server (models loaded on-demand):

```bash
embeddy serve [OPTIONS]
```

Options:
- `--device`: Device to run on (default: `cpu`, options: `cpu`, `cuda:0`, etc.)
- `--port`: Port to listen on (default: `8080`)
- `--host`: Host to bind to (default: `0.0.0.0`)

Examples:

```bash
# Start server on default port (models loaded on-demand)
embeddy serve

# Start server on custom port
embeddy serve --port 3000

# Start server with GPU
embeddy serve --device cuda:0
```

**Note**: Models are loaded automatically when first requested via the API. You don't need to specify a model at startup.

### HTTP API

#### Health Check

Check server status and loaded models:

**Request:**
```bash
curl http://localhost:8080/api/health
```

**Response:**
```json
{
  "status": "ok",
  "loaded_models": ["minilm", "mpnet"],
  "device": "Cpu"
}
```

#### Generate Embeddings

Generate embeddings for text inputs. Models are loaded automatically on first request.

**Request:**
```bash
curl -X POST http://localhost:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minilm",
    "input": [
      "Machine learning is transforming technology",
      "Natural language processing enables human-computer interaction"
    ]
  }'
```

**Request Body:**
```json
{
  "model": "minilm",
  "input": ["Text to embed", "Another text"]
}
```

**Response:**
```json
{
  "model": "minilm",
  "dimension": 384,
  "embeddings": [
    [0.123, -0.456, 0.789, ...],
    [0.321, -0.654, 0.987, ...]
  ]
}
```

**Multiple Models:**

You can use different models in the same server instance:

```bash
# Use MiniLM model
curl -X POST http://localhost:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "minilm", "input": ["Test text"]}'

# Use MPNet model (loaded automatically if not cached)
curl -X POST http://localhost:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "mpnet", "input": ["Test text"]}'
```

## Configuration

Embeddy uses environment variables for configuration:

- `EMBEDDY_DATA_DIR`: Directory for storing models and registry (default: system data directory)
- `RUST_LOG`: Logging level (default: `info`, options: `debug`, `info`, `warn`, `error`)

### Data Directory Structure

```
$EMBEDDY_DATA_DIR/
├── models/           # Downloaded model files
└── models.toml       # Model registry
```

Default locations:
- Linux: `~/.local/share/embeddy/`
- macOS: `~/Library/Application Support/embeddy/`
- Windows: `C:\Users\<USER>\AppData\Roaming\embeddy\`

## Docker Deployment

### Using Docker Compose (Recommended)

The simplest way to deploy Embeddy:

```bash
# Start the server
docker-compose up -d

# Download models inside the container
docker exec embeddy embeddy pull sentence-transformers/all-MiniLM-L6-v2 --alias minilm
docker exec embeddy embeddy pull sentence-transformers/all-mpnet-base-v2 --alias mpnet

# List installed models
docker exec embeddy embeddy list

# Use the API (models loaded on-demand)
curl -X POST http://localhost:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "minilm", "input": ["Hello, world!"]}'
```

### Using Dockerfile

Build and run manually:

```bash
# Build the image
docker build -t embeddy:latest .

# Run with volume for persistent model storage
docker run -d \
  --name embeddy \
  -p 8080:8080 \
  -v embeddy-data:/data \
  embeddy:latest

# Download models
docker exec embeddy embeddy pull sentence-transformers/all-MiniLM-L6-v2 --alias minilm
```

### Configuration

The `docker-compose.yml` includes:
- **Port mapping**: `8080:8080`
- **Persistent volume**: `embeddy-data:/data` (stores downloaded models)
- **Restart policy**: `unless-stopped`
- **Environment**: `EMBEDDY_DATA_DIR=/data`

Models are persisted in the volume and will be available after container restarts.

## Supported Models

Embeddy supports most sentence-transformer models from HuggingFace Hub. Recommended models:

| Model | Dimension | Description |
|-------|-----------|-------------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast and efficient, good for general use |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Higher quality, larger size |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual support |
| `BAAI/bge-small-en-v1.5` | 384 | Optimized for retrieval tasks |
| `BAAI/bge-base-en-v1.5` | 768 | Better quality, retrieval-focused |

### Model Requirements

- Model must include `config.json`, `tokenizer.json`, and weights file
- Supported weight formats: SafeTensors (`.safetensors`) or PyTorch (`.bin`)
- Model type: BERT-based architectures (BERT, RoBERTa, DistilBERT, etc.)

## Development

### Building from Source

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run -- serve --model minilm
```

### Project Structure

```
embeddy/
├── src/
│   ├── cli/          # CLI argument parsing
│   ├── config.rs     # Configuration management
│   ├── embedder/     # Model loading and inference
│   ├── error.rs      # Error types
│   ├── model/        # Model downloading and registry
│   ├── server/       # HTTP API server
│   └── main.rs       # Application entry point
├── Cargo.toml        # Rust dependencies
├── Dockerfile        # Docker image definition
└── docker-compose.yml
```

### Dependencies

- **candle-core**: ML inference framework
- **axum**: HTTP server
- **tokenizers**: Text tokenization
- **hf-hub**: HuggingFace model downloading
- **clap**: CLI argument parsing
- **tokio**: Async runtime

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/cedrugs/embeddy).
