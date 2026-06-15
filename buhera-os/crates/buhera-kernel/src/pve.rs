//! Proof Validation Engine (lightweight).
//!
//! Performs structural sanity checks on vaHera statements before the
//! kernel dispatches them. A full Lean 4 backend can plug in later; for
//! the working OS this is sufficient to refuse malformed or unknown
//! statement kinds.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// Error produced when a vaHera statement fails PVE.
#[derive(Debug, Clone, PartialEq, Error)]
#[error("{kind}: {reason}")]
pub struct PveError {
    /// The statement kind that failed.
    pub kind: String,
    /// Human-readable reason.
    pub reason: String,
}

/// PVE statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PveStats {
    /// Statements admitted.
    pub verified: u64,
    /// Statements rejected.
    pub rejected: u64,
}

/// Proof Validation Engine.
#[derive(Debug, Default)]
pub struct Pve {
    verified: u64,
    rejected: u64,
    events: Vec<String>,
}

impl Pve {
    /// Construct a fresh PVE.
    pub fn new() -> Self {
        Self::default()
    }

    /// Verify a vaHera statement of kind `statement_type` against its
    /// payload. Returns `Ok(())` on success, [`PveError`] otherwise.
    pub fn verify(
        &mut self,
        statement_type: &str,
        payload: &BTreeMap<String, serde_json::Value>,
    ) -> Result<(), PveError> {
        let result = match statement_type {
            "resolve" => match payload.get("target") {
                Some(v) if !v.is_null() && v.as_str().map(|s| !s.is_empty()).unwrap_or(true) => {
                    Ok(())
                }
                _ => Err("resolve requires target"),
            },
            "navigate" => match payload.get("mode").and_then(|v| v.as_str()) {
                Some("penultimate") | Some("explicit") => Ok(()),
                _ => Err("navigate requires mode"),
            },
            "complete" => {
                if payload.contains_key("s_penultimate") {
                    Ok(())
                } else {
                    Err("complete requires penultimate")
                }
            }
            "memory_create" => {
                if payload.contains_key("coord") {
                    Ok(())
                } else {
                    Err("memory create requires coord")
                }
            }
            "memory_read" | "memory_write" | "memory_store" | "memory_find" | "memory_list"
            | "memory_dump" | "demon_sort" | "controller_verify" | "kernel_stats"
            | "kernel_trace" | "process_list" | "describe" | "spawn" => Ok(()),
            other => {
                let msg = format!("unknown statement {}", other);
                self.rejected += 1;
                let evt = format!("PVE.verify {} REJECTED: {}", other, msg);
                self.events.push(evt);
                return Err(PveError {
                    kind: other.to_string(),
                    reason: msg,
                });
            }
        };

        match result {
            Ok(()) => {
                self.verified += 1;
                self.events
                    .push(format!("PVE.verify {} OK", statement_type));
                Ok(())
            }
            Err(reason) => {
                self.rejected += 1;
                self.events.push(format!(
                    "PVE.verify {} REJECTED: {}",
                    statement_type, reason
                ));
                Err(PveError {
                    kind: statement_type.to_string(),
                    reason: reason.to_string(),
                })
            }
        }
    }

    /// Cumulative statistics.
    pub fn stats(&self) -> PveStats {
        PveStats {
            verified: self.verified,
            rejected: self.rejected,
        }
    }

    /// Activity events.
    pub fn events(&self) -> Vec<String> {
        self.events.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unknown_statement_rejected() {
        let mut pve = Pve::new();
        let payload = BTreeMap::new();
        assert!(pve.verify("zzz", &payload).is_err());
    }

    #[test]
    fn resolve_requires_target() {
        let mut pve = Pve::new();
        let payload = BTreeMap::new();
        assert!(pve.verify("resolve", &payload).is_err());
        let mut payload = BTreeMap::new();
        payload.insert("target".to_string(), json!("ethanol"));
        assert!(pve.verify("resolve", &payload).is_ok());
    }

    #[test]
    fn navigate_requires_mode() {
        let mut pve = Pve::new();
        let payload = BTreeMap::new();
        assert!(pve.verify("navigate", &payload).is_err());
        let mut payload = BTreeMap::new();
        payload.insert("mode".to_string(), json!("penultimate"));
        assert!(pve.verify("navigate", &payload).is_ok());
    }
}
