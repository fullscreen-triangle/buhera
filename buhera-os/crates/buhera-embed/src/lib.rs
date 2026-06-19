//! Semantic text embedding for the Buhera OS.
//!
//! Wraps a local ONNX sentence-transformer (via `fastembed-rs`) and
//! projects its 384-dimensional output onto the three S-entropy axes
//! `(S_k, S_t, S_e)` using three fixed orthogonal directions.
//!
//! # Trait
//!
//! [`TextEmbedder`] is the abstraction shared with the lexical
//! fallback in [`buhera_substrate`]. The kernel uses
//! `Box<dyn TextEmbedder>` so it can swap implementations at boot.
//!
//! # Default implementation
//!
//! [`SemanticEmbedder`] holds a loaded `fastembed` model plus the
//! deterministic projection matrix. Construction loads the model from
//! the user's cache (downloading it on first run); subsequent calls to
//! [`SemanticEmbedder::embed`] are pure inference at the cost of a few
//! milliseconds per sentence on CPU.
//!
//! # Lexical fallback
//!
//! [`LexicalEmbedder`] re-exports the lexical hash-bag embedding from
//! [`buhera_substrate::embed_text`] behind the same trait. The kernel
//! falls back to this when no model is available or when the user
//! explicitly disables semantic mode.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod projection;
mod semantic;
mod lexical;
mod overlap;

pub use semantic::{SemanticEmbedder, SemanticConfig, EmbedError};
pub use lexical::LexicalEmbedder;
pub use overlap::token_overlap_score;

use buhera_substrate::SCoord;

/// A pluggable text embedder.
///
/// Implementations promise determinism: the same input always yields
/// the same output. They do not promise speed bounds; callers should
/// expect milliseconds per sentence for the semantic backend.
pub trait TextEmbedder: Send + Sync {
    /// Embed `text` into S-entropy space.
    fn embed(&self, text: &str) -> SCoord;

    /// Embed a batch of texts. Default impl loops over `embed`;
    /// concrete implementations may override for vectorised inference.
    fn embed_batch(&self, texts: &[&str]) -> Vec<SCoord> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Human-readable name of this embedder (for logging).
    fn name(&self) -> &'static str;
}
