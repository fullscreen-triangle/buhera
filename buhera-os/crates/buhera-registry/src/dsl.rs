//! The DSL registry (specification `04-dsl-registry.md`).
//!
//! Every Buhera module that owns a language registers it here. An entry
//! answers three questions for the generate → validate → repair loop:
//!
//! * **Is this source well-formed?** — `validate`, which must call the
//!   language's *real* front end (lexer/parser/typechecker), never a
//!   re-implementation. A script is valid iff its own compiler accepts it.
//! * **Who runs it?** — `module_id`, the registry id to dispatch the
//!   validated source to.
//! * **What grounds generation?** — `pack_id`, the knowledge pack holding
//!   the grammar reference and worked examples.
//!
//! Compilers disagree on error shape (throw, `{valid}`, `{ok}`, `Result`).
//! Each entry's validator normalises to [`Validation`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// One diagnostic from a DSL front end.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DslError {
    /// Human-readable message, as the compiler phrased it.
    pub message: String,
    /// 1-based line, when the compiler reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    /// 1-based column, when the compiler reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column: Option<u32>,
}

impl DslError {
    /// An error with no position.
    pub fn msg(message: impl Into<String>) -> Self {
        Self { message: message.into(), line: None, column: None }
    }

    /// An error at a line (and optional column).
    pub fn at(message: impl Into<String>, line: u32, column: Option<u32>) -> Self {
        Self { message: message.into(), line: Some(line), column }
    }
}

/// The normalised verdict of a DSL front end.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Validation {
    /// True iff the language's own compiler accepted the source.
    pub ok: bool,
    /// Diagnostics; empty when `ok`.
    pub errors: Vec<DslError>,
}

impl Validation {
    /// Accepted.
    pub fn valid() -> Self {
        Self { ok: true, errors: Vec::new() }
    }

    /// Rejected with these diagnostics (at least one is required).
    pub fn invalid(errors: Vec<DslError>) -> Self {
        debug_assert!(!errors.is_empty(), "an invalid verdict must carry a diagnostic");
        Self { ok: false, errors }
    }
}

/// A validator: source text in, verdict out. Must be pure and must not
/// execute the program.
pub type Validator = fn(&str) -> Validation;

/// One registered language.
#[derive(Clone)]
pub struct DslEntry {
    /// Language id (`"sbs"`, `"hfq"`, `"ndombolo"`, …).
    pub id: &'static str,
    /// Display name.
    pub label: &'static str,
    /// Conventional file extension, with the dot.
    pub extension: &'static str,
    /// Registry module that executes validated source.
    pub module_id: &'static str,
    /// Knowledge pack that grounds generation.
    pub pack_id: &'static str,
    /// The real front end, normalised.
    pub validate: Validator,
}

impl std::fmt::Debug for DslEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DslEntry")
            .field("id", &self.id)
            .field("label", &self.label)
            .field("extension", &self.extension)
            .field("module_id", &self.module_id)
            .field("pack_id", &self.pack_id)
            .finish()
    }
}

/// Serializable summary of an entry (no function pointer).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DslSummary {
    /// Language id.
    pub id: String,
    /// Display name.
    pub label: String,
    /// File extension.
    pub extension: String,
    /// Executing module.
    pub module_id: String,
    /// Grounding pack.
    pub pack_id: String,
}

/// Why a DSL lookup failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DslRegistryError {
    /// No language registered under this id — a programming error,
    /// distinct from invalid source (which is a [`Validation`]).
    #[error("unknown DSL: \"{0}\"")]
    UnknownDsl(String),
}

/// Language id → entry.
#[derive(Debug, Default, Clone)]
pub struct DslRegistry {
    entries: BTreeMap<&'static str, DslEntry>,
}

impl DslRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a language, replacing any entry with the same id.
    pub fn register(&mut self, entry: DslEntry) -> Option<DslEntry> {
        self.entries.insert(entry.id, entry)
    }

    /// Look up a language.
    pub fn get(&self, id: &str) -> Option<&DslEntry> {
        self.entries.get(id)
    }

    /// Registered language ids, sorted.
    pub fn ids(&self) -> Vec<&'static str> {
        self.entries.keys().copied().collect()
    }

    /// Serializable listing.
    pub fn list(&self) -> Vec<DslSummary> {
        self.entries
            .values()
            .map(|e| DslSummary {
                id: e.id.into(),
                label: e.label.into(),
                extension: e.extension.into(),
                module_id: e.module_id.into(),
                pack_id: e.pack_id.into(),
            })
            .collect()
    }

    /// Validate `source` against the named language's real front end.
    pub fn validate(&self, id: &str, source: &str) -> Result<Validation, DslRegistryError> {
        let entry = self.get(id).ok_or_else(|| DslRegistryError::UnknownDsl(id.to_string()))?;
        Ok((entry.validate)(source))
    }

    /// Which language owns a file extension (`".sbs"` → `"sbs"`).
    pub fn by_extension(&self, extension: &str) -> Option<&DslEntry> {
        self.entries.values().find(|e| e.extension.eq_ignore_ascii_case(extension))
    }
}

/// Pull a 1-based line number out of a message of the form `"line N: …"`
/// (the convention most house parsers use when they throw).
pub fn line_from_message(message: &str) -> Option<u32> {
    let lower = message.to_ascii_lowercase();
    let at = lower.find("line ")?;
    let digits: String = lower[at + 5..].chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}
