//! The relay wire protocol.
//!
//! Mirrors `buhera_gateway::relay::{ClientMsg, ServerMsg}` byte-for-byte —
//! duplicated rather than shared through a common crate, the same call made
//! for the ternary-address `DEPTH` constant: a client CLI has no business
//! depending on `buhera-gateway` (which would drag in axum, SQLite, argon2,
//! and every other server-only dependency) for two small enums. If the
//! gateway's shapes change, this must change with them — there is no compiler
//! check for that, only the shared JSON `kind` tag surfacing a mismatch at
//! runtime as an unparseable frame.

use serde::{Deserialize, Serialize};

/// A frame this catalyst sends to the gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClientMsg {
    /// The first frame on every connection: which paired machine this is.
    Hello {
        /// The name given at pairing time.
        name: String,
    },
    /// Sent on an interval so the gateway's liveness window has something
    /// to measure.
    Heartbeat,
    /// A run completed.
    RunResult {
        /// Correlates to the `Run` frame that requested it.
        request_id: String,
        /// One JSON value per statement that produced output.
        results: Vec<serde_json::Value>,
        /// Interpreter trace.
        trace: Vec<String>,
    },
    /// A run could not be completed.
    RunError {
        /// Correlates to the `Run` frame that requested it.
        request_id: String,
        /// Human-readable cause.
        message: String,
    },
}

/// A frame the gateway sends to this catalyst.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ServerMsg {
    /// Acknowledges a `Hello`; this connection is now registered.
    HelloAck,
    /// Dispatch a unit of work.
    Run {
        /// Correlates the eventual `RunResult`/`RunError` back to it.
        request_id: String,
        /// vaHera source to execute.
        source: String,
    },
}
