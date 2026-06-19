//! Lexical fallback embedder.
//!
//! Wraps the deterministic hash-bag embedding from `buhera-substrate`
//! so that callers can choose at boot time between semantic
//! ([`crate::SemanticEmbedder`]) and lexical operation. Useful when the
//! model is unavailable or when speed is more important than meaning.

use buhera_substrate::{embed_text, SCoord};

use crate::TextEmbedder;

/// Lexical embedder: defers to the deterministic hash-bag embedder
/// from [`buhera_substrate::embed_text`].
#[derive(Debug, Default, Clone, Copy)]
pub struct LexicalEmbedder;

impl LexicalEmbedder {
    /// Construct.
    pub fn new() -> Self {
        Self
    }
}

impl TextEmbedder for LexicalEmbedder {
    fn embed(&self, text: &str) -> SCoord {
        embed_text(text)
    }

    fn name(&self) -> &'static str {
        "lexical-fnv1a"
    }
}
