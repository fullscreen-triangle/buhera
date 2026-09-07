//! Per-account kernels for the gateway's own execution path.
//!
//! These are **not** the user's instance. The user's instance lives on their
//! own machine, holds their files, and is reached through the relay. What
//! lives here is the reduced-capability kernel that keeps a session usable
//! when that machine is asleep — scratch space, not a system of record.
//!
//! That distinction drives two decisions:
//!
//! * Kernels are held in memory and are lost on restart. Promoting them to
//!   durable storage would make the gateway a second, competing home for
//!   state that is supposed to have exactly one home, and would quietly
//!   invite the divergence problem — two kernels for one user, disagreeing.
//! * They are keyed by account, so logging in from the lab reaches the same
//!   scratch kernel as logging in from a phone. Within the degraded path,
//!   the session really does follow the user between machines.
//!
//! The user-visible consequence is stated rather than hidden: a run that
//! lands here is tagged with the reason, so nobody mistakes a degraded
//! answer for a full one.

use std::collections::HashMap;
use std::sync::Mutex;

use buhera_kernel::Kernel;
use buhera_vahera::{execute_vahera, MoleculeDatabase, NamedResult};

/// Ternary-address depth for gateway-side kernels.
///
/// Matches the REPL default so a statement behaves the same here as it does
/// on the user's own machine; a different depth would silently change the
/// addresses a query resolves to.
const DEPTH: usize = 12;

/// What a gateway-side execution produced.
#[derive(Debug)]
pub struct Output {
    /// One JSON value per statement that produced output.
    pub results: Vec<serde_json::Value>,
    /// Interpreter trace.
    pub trace: Vec<String>,
}

/// The set of live gateway-side kernels, one per account.
pub struct Sessions {
    inner: Mutex<HashMap<String, Kernel>>,
    molecules: MoleculeDatabase,
}

impl std::fmt::Debug for Sessions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.inner.lock().map(|m| m.len()).unwrap_or(0);
        f.debug_struct("Sessions").field("live", &n).finish()
    }
}

impl Default for Sessions {
    fn default() -> Self {
        Self::new()
    }
}

impl Sessions {
    /// Create an empty set.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            molecules: MoleculeDatabase::new(),
        }
    }

    /// How many accounts currently hold a gateway-side kernel.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// Whether no kernels are held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Discard an account's kernel, if it has one. Returns whether it did.
    pub fn reset(&self, account_id: &str) -> bool {
        self.inner
            .lock()
            .map(|mut m| m.remove(account_id).is_some())
            .unwrap_or(false)
    }

    /// Execute `source` against `account_id`'s kernel, creating it on first
    /// use.
    ///
    /// Uses the lexical embedder from the substrate rather than the semantic
    /// one: the semantic model downloads ~133 MB on first use and would make
    /// the first degraded request on a cold gateway hang for as long as that
    /// takes. Falling back to the gateway is already the slow path; it should
    /// not also be the unpredictable one.
    pub fn execute(&self, account_id: &str, source: &str) -> Result<Output, String> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| "session state was poisoned by an earlier panic".to_string())?;
        let kernel = guard
            .entry(account_id.to_string())
            .or_insert_with(|| Kernel::new(DEPTH));

        let ctx = execute_vahera(source, kernel, &self.molecules).map_err(|e| e.to_string())?;

        Ok(Output {
            results: ctx.results.iter().map(render).collect(),
            trace: ctx.trace,
        })
    }
}

/// Render one interpreter result as JSON for the wire.
fn render(r: &NamedResult) -> serde_json::Value {
    match r {
        NamedResult::FindHits { query, hits } => serde_json::json!({
            "kind": "hits",
            "query": query,
            "hits": hits.iter().map(|h| serde_json::json!({
                "name": h.value.metadata.get("name").and_then(|v| v.as_str()).unwrap_or("?"),
                "address": h.value.address,
                "distance": h.distance,
            })).collect::<Vec<_>>(),
        }),
        NamedResult::SortedObjects(objs) => serde_json::json!({
            "kind": "sorted",
            "objects": objs.iter().map(brief).collect::<Vec<_>>(),
        }),
        NamedResult::ObjectList(objs) => serde_json::json!({
            "kind": "list",
            "objects": objs.iter().map(brief).collect::<Vec<_>>(),
        }),
        NamedResult::Dump { name, obj } => serde_json::json!({
            "kind": "dump",
            "name": name,
            "object": obj.as_ref().map(|o| serde_json::json!({
                "address": o.address,
                "coord": { "k": o.coord.k, "t": o.coord.t, "e": o.coord.e },
                "tier": o.tier.as_str(),
                "payload": o.payload,
            })),
        }),
        NamedResult::Stats(v) => serde_json::json!({ "kind": "stats", "stats": v }),
        NamedResult::Trace(lines) => serde_json::json!({ "kind": "trace", "lines": lines }),
        NamedResult::Processes(ps) => serde_json::json!({
            "kind": "processes",
            "processes": ps.iter().map(|p| serde_json::json!({
                "pid": p.pid,
                "program": p.program_name,
                "state": p.state.as_str(),
            })).collect::<Vec<_>>(),
        }),
    }
}

fn brief(o: &buhera_kernel::MemoryObject) -> serde_json::Value {
    serde_json::json!({
        "name": o.metadata.get("name").and_then(|v| v.as_str()).unwrap_or("?"),
        "address": o.address,
        "tier": o.tier.as_str(),
        "coord": { "k": o.coord.k, "t": o.coord.t, "e": o.coord.e },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kernel_is_created_on_first_use_and_reused() {
        let s = Sessions::new();
        assert!(s.is_empty());
        s.execute("acct", "memory store \"a\" = \"hello world\"").unwrap();
        assert_eq!(s.len(), 1);
        s.execute("acct", "memory list").unwrap();
        assert_eq!(s.len(), 1, "second call must reuse the same kernel");
    }

    #[test]
    fn stored_content_survives_between_requests() {
        // This is what makes it a session rather than a calculator: the
        // scientist stores something, navigates away, and it is still there.
        let s = Sessions::new();
        s.execute("acct", "memory store \"note\" = \"the deadline is Friday\"")
            .unwrap();
        let out = s.execute("acct", "memory list").unwrap();
        let listed = serde_json::to_string(&out.results).unwrap();
        assert!(listed.contains("note"), "stored object missing: {listed}");
    }

    #[test]
    fn accounts_do_not_share_a_kernel() {
        // The isolation property. Alice storing something must not put it
        // where Bob can retrieve it.
        let s = Sessions::new();
        s.execute("alice", "memory store \"secret\" = \"alice private text\"")
            .unwrap();
        let bob = s.execute("bob", "memory list").unwrap();
        let listed = serde_json::to_string(&bob.results).unwrap();
        assert!(
            !listed.contains("secret"),
            "bob saw alice's object: {listed}"
        );
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn a_bad_statement_is_an_error_not_a_panic() {
        let s = Sessions::new();
        assert!(s.execute("acct", "this is not vahera at all ~~~").is_err());
    }

    #[test]
    fn a_failed_statement_leaves_the_session_usable() {
        let s = Sessions::new();
        s.execute("acct", "memory store \"a\" = \"first\"").unwrap();
        let _ = s.execute("acct", "!!! nonsense !!!");
        let out = s.execute("acct", "memory list").unwrap();
        let listed = serde_json::to_string(&out.results).unwrap();
        assert!(listed.contains("\"a\""), "earlier content lost: {listed}");
    }

    #[test]
    fn reset_discards_only_the_named_account() {
        let s = Sessions::new();
        s.execute("alice", "memory store \"x\" = \"one\"").unwrap();
        s.execute("bob", "memory store \"y\" = \"two\"").unwrap();
        assert!(s.reset("alice"));
        assert!(!s.reset("alice"), "second reset has nothing to discard");
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn retrieval_finds_what_was_stored() {
        let s = Sessions::new();
        s.execute("acct", "memory store \"lunch\" = \"pizza for lunch today\"")
            .unwrap();
        s.execute("acct", "memory store \"task\" = \"update the project readme\"")
            .unwrap();
        let out = s
            .execute("acct", "memory find nearest \"food\" k=2")
            .unwrap();
        let rendered = serde_json::to_string(&out.results).unwrap();
        assert!(rendered.contains("\"kind\":\"hits\""), "{rendered}");
    }
}
