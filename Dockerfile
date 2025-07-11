# Dockerfile for Buhera Virtual Processor Architecture
# Multi-stage build optimized for scientific computing and theoretical framework development

# Base image with Rust and scientific computing tools
FROM rust:1.75-slim-bullseye as builder

# Set environment variables
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.75.0

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    pkg-config \
    # Scientific computing libraries
    libopenblas-dev \
    liblapack-dev \
    libfftw3-dev \
    libgsl-dev \
    # Quantum computing dependencies
    libquantum-dev \
    libeigen3-dev \
    # Molecular simulation dependencies
    libhdf5-dev \
    libnetcdf-dev \
    # Neural network dependencies
    libopencv-dev \
    libtorch-dev \
    # LaTeX for documentation
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    # Additional tools
    git \
    curl \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install additional Rust components
RUN rustup component add \
    rustfmt \
    clippy \
    rust-src \
    && rustup target add \
    x86_64-unknown-linux-gnu \
    aarch64-unknown-linux-gnu \
    x86_64-apple-darwin \
    aarch64-apple-darwin

# Install cargo tools for development
RUN cargo install \
    cargo-audit \
    cargo-deny \
    cargo-outdated \
    cargo-tree \
    cargo-watch \
    cargo-expand \
    cargo-asm \
    cargo-edit \
    cargo-udeps \
    cargo-machete \
    cargo-bloat \
    cargo-benchcmp \
    cargo-criterion \
    cargo-tarpaulin \
    cargo-nextest \
    cargo-llvm-cov \
    bacon \
    just \
    taplo-cli \
    typos-cli

# Set working directory
WORKDIR /usr/src/buhera

# Copy dependency files
COPY Cargo.toml Cargo.lock ./
COPY rustfmt.toml clippy.toml ./
COPY .cargo/ .cargo/

# Create dummy source to cache dependencies
RUN mkdir -p src/ && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn lib() {}" > src/lib.rs

# Build dependencies
RUN cargo build --release && \
    cargo build && \
    rm -rf src/

# Copy source code
COPY src/ src/
COPY etc/ etc/
COPY docs/ docs/
COPY boot/ boot/
COPY opt/ opt/
COPY README.md LICENSE ./

# Build the application
RUN cargo build --release

# Final stage - runtime image
FROM debian:bullseye-slim as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    # Runtime libraries for scientific computing
    libopenblas-base \
    liblapack3 \
    libfftw3-double3 \
    libgsl25 \
    # Quantum computing runtime
    libquantum8 \
    # Molecular simulation runtime
    libhdf5-103 \
    libnetcdf18 \
    # Neural network runtime
    libopencv-core4.5 \
    # System tools
    ca-certificates \
    curl \
    # Process monitoring
    htop \
    strace \
    # Network tools
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r buhera && useradd -r -g buhera buhera

# Set working directory
WORKDIR /app

# Copy built application
COPY --from=builder /usr/src/buhera/target/release/buhera /app/buhera

# Copy configuration files
COPY --from=builder /usr/src/buhera/etc/ /app/etc/
COPY --from=builder /usr/src/buhera/opt/ /app/opt/
COPY --from=builder /usr/src/buhera/boot/ /app/boot/
COPY --from=builder /usr/src/buhera/docs/ /app/docs/
COPY --from=builder /usr/src/buhera/README.md /app/
COPY --from=builder /usr/src/buhera/LICENSE /app/

# Set up environment variables
ENV BUHERA_LOG_LEVEL=info \
    BUHERA_ENVIRONMENT=production \
    BUHERA_CONFIG_PATH=/app/etc/vpos/vpos.conf \
    RUST_LOG=info \
    RUST_BACKTRACE=1 \
    # Molecular foundry configuration
    MOLECULAR_FOUNDRY_SIMULATION=true \
    MOLECULAR_FOUNDRY_PRECISION=high \
    # Quantum coherence configuration
    QUANTUM_COHERENCE_SIMULATION=true \
    QUANTUM_COHERENCE_TEMPERATURE=room_temperature \
    # Neural network configuration
    NEURAL_NETWORK_SIMULATION=true \
    # Semantic processing configuration
    SEMANTIC_PROCESSING_MODE=cross_modal \
    # Fuzzy logic configuration
    FUZZY_DIGITAL_MODE=continuous \
    # BMD information catalysis configuration
    BMD_CATALYSIS_MODE=entropy_reduction \
    # VPOS kernel configuration
    VPOS_KERNEL_MODE=virtual_processor \
    # Masunda Temporal Navigator configuration
    MASUNDA_MEMORIAL_MODE=stella_lorraine

# Create directories for runtime data
RUN mkdir -p /app/data /app/logs /app/tmp && \
    chown -R buhera:buhera /app

# Switch to non-root user
USER buhera

# Expose ports for various services
EXPOSE 8080 8081 8082 8083 8084 8085 8086 8087 8088 8089

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["./buhera", "--config", "/app/etc/vpos/vpos.conf"]

# Development stage for debugging
FROM builder as development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    # Debugging tools
    gdb \
    valgrind \
    # Development utilities
    vim \
    tmux \
    zsh \
    # Additional scientific tools
    python3 \
    python3-pip \
    jupyter \
    # LaTeX for documentation development
    texlive-full \
    && rm -rf /var/lib/apt/lists/*

# Install Python scientific computing packages
RUN pip3 install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    scikit-learn \
    tensorflow \
    torch \
    jupyter \
    notebook \
    jupyterlab

# Set working directory
WORKDIR /usr/src/buhera

# Copy all files for development
COPY . .

# Environment variables for development
ENV BUHERA_LOG_LEVEL=debug \
    BUHERA_ENVIRONMENT=development \
    RUST_LOG=debug \
    RUST_BACKTRACE=full \
    # Development-specific debugging
    MOLECULAR_FOUNDRY_DEBUG=true \
    QUANTUM_COHERENCE_DEBUG=true \
    NEURAL_NETWORK_DEBUG=true \
    SEMANTIC_DEBUG=true \
    FUZZY_DEBUG=true \
    BMD_DEBUG=true \
    VPOS_DEBUG=true \
    MASUNDA_DEBUG=true

# Default command for development
CMD ["cargo", "run"]

# Testing stage for CI/CD
FROM builder as testing

# Copy source code and test files
COPY . .

# Run tests
RUN cargo test --all-features --release

# Run quality checks
RUN cargo fmt --all -- --check && \
    cargo clippy --all-targets --all-features -- -D warnings

# Generate test coverage
RUN cargo tarpaulin --all-features --out xml --output-dir coverage/

# Benchmark stage for performance testing
FROM builder as benchmarking

# Copy source code
COPY . .

# Run benchmarks
RUN cargo bench --all-features

# Generate performance reports
RUN cargo benchcmp --output-format json > benchmarks/results.json

# Documentation stage for generating docs
FROM builder as documentation

# Install additional documentation tools
RUN apt-get update && apt-get install -y \
    pandoc \
    graphviz \
    plantuml \
    && rm -rf /var/lib/apt/lists/*

# Copy source code and documentation
COPY . .

# Generate Rust documentation
RUN cargo doc --document-private-items --no-deps

# Build LaTeX documentation
RUN find docs -name "*.tex" -exec pdflatex -output-directory=docs/build {} \;

# Generate additional documentation formats
RUN pandoc README.md -o docs/README.pdf && \
    pandoc README.md -o docs/README.html

# Production stage optimized for deployment
FROM runtime as production

# Additional production optimizations
RUN apt-get update && apt-get install -y \
    # Monitoring tools
    collectd \
    prometheus-node-exporter \
    # Logging tools
    rsyslog \
    logrotate \
    && rm -rf /var/lib/apt/lists/*

# Production-specific environment variables
ENV BUHERA_LOG_LEVEL=warn \
    BUHERA_ENVIRONMENT=production \
    RUST_LOG=warn \
    RUST_BACKTRACE=0

# Configure logging
COPY docker/rsyslog.conf /etc/rsyslog.conf
COPY docker/logrotate.conf /etc/logrotate.d/buhera

# Set resource limits
USER root
RUN echo "buhera soft nofile 65536" >> /etc/security/limits.conf && \
    echo "buhera hard nofile 65536" >> /etc/security/limits.conf && \
    echo "buhera soft nproc 32768" >> /etc/security/limits.conf && \
    echo "buhera hard nproc 32768" >> /etc/security/limits.conf

USER buhera

# Default production command
CMD ["./buhera", "--config", "/app/etc/vpos/vpos.conf", "--production"] 