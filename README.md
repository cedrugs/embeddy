# embeddy

[![Rust](https://img.shields.io/badge/rust-1.91%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/github/actions/workflow/status/cedrugs/embeddy/ci.yml?branch=main)](https://github.com/cedrugs/embeddy/actions)

A lightweight, embeddings-only model runtime with CLI and HTTP API. Built in Rust for performance and efficiency, embeddy allows you to download, manage, and run text embedding models from HuggingFace without heavy dependencies.

## Features

- **Lightweight Runtime**: Pure Rust implementation using the Candle ML framework
- **HuggingFace Integration**: Download and cache models directly from HuggingFace Hub
- **Dynamic Model Loading**: Load multiple models on-demand via API without restart
- **Model Preloading**: Optionally preload models at startup for zero cold-start latency
- **Embedding Cache**: LRU cache for repeated queries (configurable size)
- **Auto-Download**: Automatically download models on first use
- **API Documentation**: Built-in Swagger UI at `/docs`
- **Hardware Support**: CPU and CUDA GPU acceleration
- **Model Management**: Built-in registry for tracking, aliasing, and removing models
- **Docker Ready**: Includes Dockerfile and docker-compose configurations
- **Multiple Formats**: Supports both SafeTensors and PyTorch model formats (F16/F32)

## Quick Start

### Using Docker (Recommended)

```bash
# Pull and run from GitHub Container Registry
docker pull ghcr.io/cedrugs/embeddy:latest

# Or use docker-compose for production
curl -O https://raw.githubusercontent.com/cedrugs/embeddy/main/docker-compose.prod.yml
docker-compose -f docker-compose.prod.yml up -d

# Test the API
curl -X POST http://localhost:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "sentence-transformers/all-MiniLM-L6-v2", "input": ["Hello, world!"]}'

# View API docs
open http://localhost:8080/docs
```

### From Source

```bash
# Clone and build
git clone https://github.com/cedrugs/embeddy.git
cd embeddy
cargo build --release

# Pull a model and run
./target/release/embeddy pull sentence-transformers/all-MiniLM-L6-v2 --alias minilm
./target/release/embeddy serve --model minilm
```

## Installation

### From Source

Requires Rust 1.91 or higher.

```bash
git clone https://github.com/cedrugs/embeddy.git
cd embeddy
cargo build --release
```

### Using Docker

```bash
# From GitHub Container Registry
docker pull ghcr.io/cedrugs/embeddy:latest

# Or build locally
docker build -t embeddy:latest .
```

## CLI Commands

### Pull a Model

```bash
embeddy pull <MODEL_REPO_ID> [--alias <ALIAS>]

# Examples
embeddy pull sentence-transformers/all-MiniLM-L6-v2
embeddy pull sentence-transformers/all-mpnet-base-v2 --alias mpnet
```

### List Models

```bash
embeddy list
```

### Remove a Model

```bash
embeddy remove <MODEL_NAME>

# Example
embeddy remove minilm
```

### Run Embeddings (CLI)

```bash
embeddy run <MODEL_NAME> --text <TEXT> [--text <TEXT>...] [--device <DEVICE>]

# Examples
embeddy run minilm --text "Hello world"
embeddy run minilm --text "First" --text "Second" --device cuda:0
```

### Serve HTTP API

```bash
embeddy serve [OPTIONS]
```

Options:
| Flag | Env Variable | Default | Description |
|------|--------------|---------|-------------|
| `--host` | `EMBEDDY_HOST` | `0.0.0.0` | Host to bind to |
| `--port` | `EMBEDDY_PORT` | `8080` | Port to listen on |
| `--device` | `EMBEDDY_DEVICE` | `cpu` | Device (`cpu`, `cuda:0`, etc.) |
| `--model` | `EMBEDDY_MODEL` | - | Model to preload (auto-downloads if needed) |
| `--cache-size` | `EMBEDDY_CACHE_SIZE` | `10000` | Embedding cache size |

Examples:

```bash
# Basic server (models loaded on-demand)
embeddy serve

# Preload a model at startup
embeddy serve --model minilm

# With GPU and larger cache
embeddy serve --model minilm --device cuda:0 --cache-size 50000

# Using environment variables
EMBEDDY_MODEL=minilm EMBEDDY_CACHE_SIZE=50000 embeddy serve
```

## HTTP API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/docs` | Swagger UI documentation |
| GET | `/openapi.json` | OpenAPI specification |
| GET | `/api/health` | Health check and status |
| POST | `/api/embed` | Generate embeddings |
| GET | `/api/models` | List installed models |
| DELETE | `/api/models/{name}` | Remove a model |

### Generate Embeddings

```bash
curl -X POST http://localhost:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minilm",
    "input": ["Hello, world!", "How are you?"]
  }'
```

Response:
```json
{
  "model": "minilm",
  "dimension": 384,
  "embeddings": [[0.123, -0.456, ...], [0.321, -0.654, ...]],
  "cache_hits": 0
}
```

### Health Check

```bash
curl http://localhost:8080/api/health
```

Response:
```json
{
  "status": "ok",
  "loaded_models": ["minilm"],
  "cache_entries": 42,
  "device": "Cpu"
}
```

### List Models

```bash
curl http://localhost:8080/api/models
```

### Remove a Model

```bash
curl -X DELETE http://localhost:8080/api/models/minilm
```

## Docker Deployment

### Development (build from source)

```bash
docker-compose up -d
```

### Production (from ghcr.io)

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Environment Variables

Configure via environment in `docker-compose.prod.yml`:

```yaml
environment:
  - EMBEDDY_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Auto-downloads and preloads
  - EMBEDDY_CACHE_SIZE=10000
  - EMBEDDY_DEVICE=cpu
  - RUST_LOG=info
```

If `EMBEDDY_MODEL` is set to a model that isn't installed, it will be automatically downloaded from HuggingFace on startup.

### Persistent Storage

Models are stored in a Docker volume (`embeddy-data:/data`) and persist across container restarts.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDY_DATA_DIR` | System default | Data directory for models |
| `EMBEDDY_HOST` | `0.0.0.0` | Server host |
| `EMBEDDY_PORT` | `8080` | Server port |
| `EMBEDDY_DEVICE` | `cpu` | Compute device |
| `EMBEDDY_MODEL` | - | Model to preload |
| `EMBEDDY_CACHE_SIZE` | `10000` | LRU cache size |
| `RUST_LOG` | `info` | Log level |

### Data Directory

Default locations:
- Linux: `~/.local/share/embeddy/`
- macOS: `~/Library/Application Support/embeddy/`
- Windows: `C:\Users\<USER>\AppData\Roaming\embeddy\`

## Supported Models

| Model | Dimension | Description |
|-------|-----------|-------------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast, good for general use |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Higher quality |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual |
| `intfloat/multilingual-e5-large-instruct` | 1024 | High quality multilingual |
| `BAAI/bge-small-en-v1.5` | 384 | Retrieval optimized |
| `BAAI/bge-base-en-v1.5` | 768 | Retrieval optimized |

### Requirements

- Must include `config.json`, `tokenizer.json`, and weights file
- Supported formats: SafeTensors (`.safetensors`) or PyTorch (`.bin`)
- Supported dtypes: F16, F32 (auto-converted)
- Architecture: BERT-based (BERT, RoBERTa, DistilBERT, etc.)

## Development

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run -- serve --model minilm
```

### Project Structure

```
embeddy/
├── src/
│   ├── cli/          # CLI argument parsing
│   ├── config.rs     # Configuration
│   ├── embedder/     # Model loading and inference
│   ├── error.rs      # Error types
│   ├── model/        # Model downloading and registry
│   ├── server/       # HTTP API server
│   └── main.rs       # Entry point
├── Cargo.toml
├── Dockerfile
├── docker-compose.yml       # Development
└── docker-compose.prod.yml  # Production
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/cedrugs/embeddy).
