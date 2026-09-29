//! The relay: a persistent connection a catalyst dials out over.
//!
//! Every other doc comment in this crate ([`crate::lib`], [`crate::router`],
//! [`crate::session`], [`crate::token`], `store::Catalyst`) already describes
//! the shape this fills in: the gateway never dials the user's machine — it
//! sits behind NAT and cannot accept inbound connections — so the machine
//! dials out, holds the socket open, and the gateway dispatches `/api/run`
//! work over it instead of running it on the degraded, gateway-side kernel.
//!
//! What lives here is purely in-memory, exactly like [`crate::session::Sessions`]
//! and for the same reason: a connection handle cannot be serialized, has
//! no meaning after a restart, and promoting it to the [`crate::store::Store`]
//! would make durable state out of something that is, definitionally,
//! ephemeral.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

/// How long the gateway waits for a dispatched run to come back before
/// giving up on that specific request.
///
/// Bounded so a catalyst that accepted the connection but then hangs (a
/// stuck kernel, a frozen process) cannot hold an HTTP request open
/// indefinitely — the caller gets a clear error instead of a silent stall.
pub const RUN_TIMEOUT: Duration = Duration::from_secs(30);

/// A frame the catalyst sends.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClientMsg {
    /// The first frame on every connection: which catalyst this is.
    ///
    /// The catalyst token alone only proves the *account* — one account can
    /// have several paired machines sharing the same token audience type —
    /// so the socket has to separately assert which named catalyst it is.
    Hello {
        /// The catalyst's name, as given at pairing time.
        name: String,
    },
    /// Sent on an interval so the gateway's liveness window
    /// ([`crate::router::LIVENESS_WINDOW_SECS`]) has something to measure.
    Heartbeat,
    /// A run completed. Carries the same shape [`crate::session::Output`]
    /// does, so the gateway can render it identically regardless of where
    /// it executed.
    RunResult {
        /// Correlates to the `Run` frame that requested it.
        request_id: String,
        /// One JSON value per statement that produced output.
        results: Vec<serde_json::Value>,
        /// Interpreter trace.
        trace: Vec<String>,
    },
    /// A run could not be completed — a bad statement, not a transport
    /// failure (a dropped socket is detected by the gateway itself).
    RunError {
        /// Correlates to the `Run` frame that requested it.
        request_id: String,
        /// Human-readable cause.
        message: String,
    },
}

/// A frame the gateway sends.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ServerMsg {
    /// Acknowledges a `Hello`; the catalyst is now registered and can be
    /// dispatched to.
    HelloAck,
    /// Dispatch a unit of work.
    Run {
        /// Correlates the eventual `RunResult`/`RunError` back to the
        /// HTTP request that is waiting on it.
        request_id: String,
        /// vaHera source to execute.
        source: String,
    },
}

/// One connected catalyst's write half, plus whatever runs are in flight
/// on it.
struct Connection {
    /// Send a frame to this catalyst. Closes when the connection task
    /// exits, which is how a dispatch attempt against a dead socket fails
    /// fast instead of hanging.
    outbound: mpsc::UnboundedSender<ServerMsg>,
    /// Requests sent but not yet answered, keyed by `request_id`. Each
    /// dispatch installs one; the connection task's read loop resolves it
    /// on the matching `RunResult`/`RunError` and removes it either way.
    pending: HashMap<String, oneshot::Sender<Result<RunOutcome, String>>>,
}

/// What a relayed run produced — the catalyst-side counterpart of
/// [`crate::session::Output`].
#[derive(Debug)]
pub struct RunOutcome {
    /// One JSON value per statement that produced output.
    pub results: Vec<serde_json::Value>,
    /// Interpreter trace.
    pub trace: Vec<String>,
}

/// Why a dispatch to a catalyst could not be completed.
#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    /// No connection is registered for this `(account_id, name)` — it was
    /// never live, or disconnected between the routing decision and the
    /// dispatch attempt.
    #[error("catalyst {name:?} is not connected")]
    NotConnected {
        /// The catalyst's name.
        name: String,
    },
    /// The catalyst accepted the request but its own kernel rejected the
    /// source.
    #[error("{0}")]
    Remote(String),
    /// No `RunResult`/`RunError` arrived within [`RUN_TIMEOUT`].
    #[error("catalyst {name:?} did not respond within {secs}s")]
    Timeout {
        /// The catalyst's name.
        name: String,
        /// The timeout that elapsed, in seconds, for the message.
        secs: u64,
    },
    /// The connection closed while a request was in flight.
    #[error("catalyst {name:?} disconnected while the request was in flight")]
    Disconnected {
        /// The catalyst's name.
        name: String,
    },
}

/// The set of live catalyst connections, across every account.
///
/// Keyed by `(account_id, name)` rather than by name alone: two different
/// accounts are free to each name a machine "laptop", and nothing here
/// should have to know that at the type level — the account id is already
/// the isolation boundary everywhere else in this crate.
#[derive(Default)]
pub struct Relay {
    connections: Mutex<HashMap<(String, String), Connection>>,
}

impl std::fmt::Debug for Relay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.connections.lock().map(|m| m.len()).unwrap_or(0);
        f.debug_struct("Relay").field("connected", &n).finish()
    }
}

impl Relay {
    /// An empty relay, no catalysts connected.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many catalysts are currently connected, across all accounts.
    pub fn len(&self) -> usize {
        self.connections.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// Whether no catalyst is connected.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Register a freshly connected catalyst, replacing any prior
    /// connection under the same key.
    ///
    /// A prior connection being replaced (rather than rejected) matches
    /// what a real client does: a machine that reconnects after a network
    /// blip should win, not be locked out by its own stale socket. The old
    /// connection's `outbound` sender is simply dropped; its read loop
    /// will notice the send side closing (or its own socket erroring) and
    /// exit on its own.
    pub fn register(&self, account_id: String, name: String, outbound: mpsc::UnboundedSender<ServerMsg>) {
        let mut guard = self.connections.lock().unwrap_or_else(|e| e.into_inner());
        guard.insert(
            (account_id, name),
            Connection { outbound, pending: HashMap::new() },
        );
    }

    /// Deregister a catalyst — called when its connection task exits, for
    /// any reason. A no-op if a newer connection already replaced it
    /// (matched by comparing the sender, so an old task's cleanup cannot
    /// evict a connection that has already been superseded).
    pub fn deregister(&self, account_id: &str, name: &str, outbound: &mpsc::UnboundedSender<ServerMsg>) {
        let mut guard = self.connections.lock().unwrap_or_else(|e| e.into_inner());
        let key = (account_id.to_string(), name.to_string());
        if let Some(conn) = guard.get(&key) {
            if conn.outbound.same_channel(outbound) {
                guard.remove(&key);
            }
        }
    }

    /// Whether a catalyst is currently connected.
    pub fn is_connected(&self, account_id: &str, name: &str) -> bool {
        let guard = self.connections.lock().unwrap_or_else(|e| e.into_inner());
        guard.contains_key(&(account_id.to_string(), name.to_string()))
    }

    /// Dispatch `source` to a connected catalyst and wait for its answer.
    ///
    /// Returns [`DispatchError::NotConnected`] immediately if there is no
    /// registered connection — the caller (`http::run`) should fall back
    /// to reporting that rather than inventing a retry, since silently
    /// running the work elsewhere would answer against the wrong
    /// filesystem, exactly the invariant the pre-relay code already
    /// enforced.
    pub async fn dispatch(&self, account_id: &str, name: &str, source: &str) -> Result<RunOutcome, DispatchError> {
        let request_id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel();

        {
            let mut guard = self.connections.lock().unwrap_or_else(|e| e.into_inner());
            let key = (account_id.to_string(), name.to_string());
            let conn = guard.get_mut(&key).ok_or_else(|| DispatchError::NotConnected { name: name.to_string() })?;
            conn.pending.insert(request_id.clone(), tx);
            if conn
                .outbound
                .send(ServerMsg::Run { request_id: request_id.clone(), source: source.to_string() })
                .is_err()
            {
                // The send side is closed — the connection task has already
                // exited but has not yet deregistered itself. Treat it the
                // same as not being connected at all.
                conn.pending.remove(&request_id);
                guard.remove(&key);
                return Err(DispatchError::NotConnected { name: name.to_string() });
            }
        }

        match tokio::time::timeout(RUN_TIMEOUT, rx).await {
            Ok(Ok(Ok(outcome))) => Ok(outcome),
            Ok(Ok(Err(message))) => Err(DispatchError::Remote(message)),
            Ok(Err(_)) => Err(DispatchError::Disconnected { name: name.to_string() }),
            Err(_) => {
                // Timed out: stop waiting on this request. If the answer
                // arrives after this, resolving an already-dropped sender
                // is a harmless no-op in the read loop.
                let mut guard = self.connections.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(conn) = guard.get_mut(&(account_id.to_string(), name.to_string())) {
                    conn.pending.remove(&request_id);
                }
                Err(DispatchError::Timeout { name: name.to_string(), secs: RUN_TIMEOUT.as_secs() })
            }
        }
    }

    /// Resolve a pending request with the catalyst's answer.
    ///
    /// Called by the connection's read loop on `RunResult`/`RunError`. A
    /// missing `request_id` (already timed out, or a stray/duplicate
    /// frame) is silently ignored — there is nothing left to resolve.
    fn resolve(&self, account_id: &str, name: &str, request_id: &str, outcome: Result<RunOutcome, String>) {
        let mut guard = self.connections.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(conn) = guard.get_mut(&(account_id.to_string(), name.to_string())) {
            if let Some(tx) = conn.pending.remove(request_id) {
                let _ = tx.send(outcome);
            }
        }
    }
}

/// Resolve a `RunResult` or `RunError` frame against `relay`.
///
/// A free function rather than a `Relay` method taking `ClientMsg`
/// directly, so the WebSocket handler in [`crate::http`] can match on the
/// full `ClientMsg` enum itself (to also handle `Hello`/`Heartbeat`)
/// without this module needing to know about connection setup.
pub fn resolve_client_msg(relay: &Relay, account_id: &str, name: &str, msg: ClientMsg) {
    match msg {
        ClientMsg::RunResult { request_id, results, trace } => {
            relay.resolve(account_id, name, &request_id, Ok(RunOutcome { results, trace }));
        }
        ClientMsg::RunError { request_id, message } => {
            relay.resolve(account_id, name, &request_id, Err(message));
        }
        ClientMsg::Hello { .. } | ClientMsg::Heartbeat => {
            // Handled by the connection task itself (registration and
            // liveness touch respectively) before this function is called.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn dispatch_to_an_unconnected_catalyst_fails_fast() {
        let relay = Relay::new();
        let err = relay.dispatch("acct", "office", "memory list").await.unwrap_err();
        assert!(matches!(err, DispatchError::NotConnected { .. }));
    }

    #[tokio::test]
    async fn a_registered_catalyst_can_be_dispatched_to_and_answers() {
        let relay = std::sync::Arc::new(Relay::new());
        let (tx, mut rx) = mpsc::unbounded_channel();
        relay.register("acct".into(), "office".into(), tx);
        assert!(relay.is_connected("acct", "office"));

        let relay_for_task = relay.clone();
        let task = tokio::spawn(async move { relay_for_task.dispatch("acct", "office", "memory list").await });

        // Stand in for the connection task's read loop: receive the Run
        // frame, then resolve it as that loop would on a RunResult.
        let ServerMsg::Run { request_id, .. } = rx.recv().await.expect("a Run frame was sent") else {
            panic!("expected a Run frame");
        };
        resolve_client_msg(
            &relay,
            "acct",
            "office",
            ClientMsg::RunResult {
                request_id,
                results: vec![serde_json::json!({"ok": true})],
                trace: vec!["done".into()],
            },
        );

        let outcome = task.await.unwrap().unwrap();
        assert_eq!(outcome.results.len(), 1);
        assert_eq!(outcome.trace, vec!["done".to_string()]);
    }

    #[tokio::test]
    async fn deregister_ignores_a_superseded_connection() {
        let relay = Relay::new();
        let (tx1, _rx1) = mpsc::unbounded_channel();
        let (tx2, _rx2) = mpsc::unbounded_channel();
        relay.register("acct".into(), "office".into(), tx1.clone());
        relay.register("acct".into(), "office".into(), tx2.clone());
        // tx1's connection task cleaning up must not evict tx2's live one.
        relay.deregister("acct", "office", &tx1);
        assert!(relay.is_connected("acct", "office"));
        relay.deregister("acct", "office", &tx2);
        assert!(!relay.is_connected("acct", "office"));
    }
}
