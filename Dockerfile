# Multi-stage build for minimal final image
FROM rust:1.91-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
	pkg-config \
	libssl-dev \
	g++ \
	&& rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml ./

# Copy source code
COPY src ./src

# Build release binary
RUN cargo build --release

# Final stage - minimal runtime image
FROM debian:bookworm-slim

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
	ca-certificates \
	libssl3 \
	&& rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 embeddy

# Copy binary from builder
COPY --from=builder /build/target/release/embeddy /usr/local/bin/embeddy

# Create data directory
RUN mkdir -p /data && chown embeddy:embeddy /data

USER embeddy
WORKDIR /home/embeddy

# Set environment variable for data directory
ENV EMBEDDY_DATA_DIR=/data
ENV RUST_LOG=info

EXPOSE 8080

ENTRYPOINT ["embeddy"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
