//! Semantic embedder backed by a local ONNX sentence-transformer.

use std::path::PathBuf;

use buhera_substrate::SCoord;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use thiserror::Error;

use crate::projection::Projection;
use crate::TextEmbedder;

/// Error type for [`SemanticEmbedder`] construction and inference.
#[derive(Debug, Error)]
pub enum EmbedError {
    /// Failed to load the underlying model.
    #[error("failed to load embedding model: {0}")]
    ModelLoad(String),
    /// Inference failed at runtime.
    #[error("inference failed: {0}")]
    Inference(String),
}

/// Configuration for [`SemanticEmbedder`].
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    /// Which model to load. Default: `BGESmallENV15Q` (quantised
    /// BGE-small-en-v1.5, ~33 MB). This model was specifically trained
    /// for retrieval and outperforms MiniLM on synonym and paraphrase
    /// queries while keeping the embedding dimension at 384 (so the
    /// rest of the projection machinery works unchanged).
    pub model: EmbeddingModel,
    /// Optional override for the cache directory. When `None`, fastembed
    /// uses its default location.
    pub cache_dir: Option<PathBuf>,
    /// Whether to show download progress on stdout. Off by default.
    pub show_download_progress: bool,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            // Full-precision BGE-small. The quantised variant has a
            // known fastembed-v4.9 issue (missing LayerNorm
            // initialisers); use the full model instead. ~133 MB on
            // disk after download; ~0.5 s per query on CPU.
            model: EmbeddingModel::BGESmallENV15,
            cache_dir: None,
            show_download_progress: false,
        }
    }
}

/// A semantic text embedder backed by a sentence-transformer model.
pub struct SemanticEmbedder {
    model: TextEmbedding,
    projection: Projection,
    name: &'static str,
}

impl SemanticEmbedder {
    /// Load the default semantic embedder (BGE-small-en-v1.5, quantised).
    ///
    /// The first call downloads the model (~33 MB) to fastembed's cache.
    /// Subsequent calls reuse the cached files.
    pub fn new() -> Result<Self, EmbedError> {
        Self::with_config(SemanticConfig::default())
    }

    /// Load with an explicit configuration.
    pub fn with_config(cfg: SemanticConfig) -> Result<Self, EmbedError> {
        let name = name_for_model(&cfg.model);
        let mut opts = InitOptions::new(cfg.model.clone())
            .with_show_download_progress(cfg.show_download_progress);
        if let Some(dir) = cfg.cache_dir {
            opts = opts.with_cache_dir(dir);
        }

        let model = TextEmbedding::try_new(opts)
            .map_err(|e| EmbedError::ModelLoad(e.to_string()))?;

        Ok(Self {
            model,
            projection: Projection::default(),
            name,
        })
    }
}

fn name_for_model(m: &EmbeddingModel) -> &'static str {
    match m {
        EmbeddingModel::BGESmallENV15Q => "bge-small-en-v1.5",
        EmbeddingModel::BGESmallENV15 => "bge-small-en-v1.5",
        EmbeddingModel::BGEBaseENV15Q => "bge-base-en-v1.5",
        EmbeddingModel::BGEBaseENV15 => "bge-base-en-v1.5",
        EmbeddingModel::AllMiniLML6V2Q => "all-MiniLM-L6-v2",
        EmbeddingModel::AllMiniLML6V2 => "all-MiniLM-L6-v2",
        _ => "fastembed-model",
    }
}

impl TextEmbedder for SemanticEmbedder {
    fn embed(&self, text: &str) -> SCoord {
        if text.trim().is_empty() {
            return SCoord::origin();
        }
        match self.model.embed(vec![text], None) {
            Ok(mut vs) => vs.pop().map(|v| self.projection.project(&v)).unwrap_or_else(SCoord::origin),
            Err(_) => SCoord::origin(),
        }
    }

    fn embed_batch(&self, texts: &[&str]) -> Vec<SCoord> {
        if texts.is_empty() {
            return Vec::new();
        }
        match self.model.embed(texts.to_vec(), None) {
            Ok(vs) => vs.iter().map(|v| self.projection.project(v)).collect(),
            Err(_) => texts.iter().map(|_| SCoord::origin()).collect(),
        }
    }

    fn name(&self) -> &'static str {
        self.name
    }
}
