//! Shim: the `.purpose/index.json` schema, copied from upstream
//! `tracker/src/purpose.rs` (the `Symbol` and `Index` structs only). The
//! upstream module's process-spawning bridge to the `purpose` CLI is
//! deliberately not vendored — the Buhera adapter is handed the index.
use serde::Deserialize;

/// One indexed definition/heading — a vertex of the repo self-graph.
#[derive(Debug, Clone, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub kind: String,
    pub file: String,
    pub line: usize,
    #[serde(default)]
    pub snippet: String,
}

/// The parsed `.purpose/index.json` — the repo's self-graph, owned by `purpose`.
#[derive(Debug, Clone, Deserialize)]
pub struct Index {
    pub root: String,
    pub symbols: Vec<Symbol>,
}
