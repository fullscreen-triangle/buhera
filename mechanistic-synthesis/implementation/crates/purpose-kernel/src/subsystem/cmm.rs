//! CMM — Categorical Memory Manager.
//!
//! Grounded in UTL Theorem 5 (Cache Extinction): when a decision class becomes
//! categorically constant — e.g. when a content-addressed lookup collapses an
//! entire compute path to a single fetch — the corresponding decision lag
//! $\tau$ vanishes structurally rather than asymptotically. The CMM is the
//! component that realises this collapse.
//!
//! Phase 1 ships a no-op CMM that always reports "miss" and never inserts.
//! Phase 3 implements the argument-hash cache and exposes per-class extinction
//! events as observable metrics.

use std::collections::BTreeMap;

use purpose_core::{Value, VaHera};

use crate::subsystem::Subsystem;

/// Cache-lookup result.
///
/// Phase 1 only ever returns [`Lookup::Miss`]. Phase 3 introduces hits and
/// records the extinction event on the kernel's observable.
#[derive(Debug, Clone)]
pub enum Lookup {
    /// No entry; the dispatch must proceed to the executor.
    Miss,
    /// A previously-cached value for the same `(op, args)` pair.
    Hit(Value),
}

/// Per-dispatch lookup and insertion hooks. Both are invoked from the kernel
/// dispatch path, so they must be cheap.
#[async_trait::async_trait]
pub trait Cmm: Subsystem {
    /// Lookup the result for an `op` invocation with the canonicalised
    /// `args`. The kernel only consults the CMM for [`VaHera::Call`] roots;
    /// `Compose` and `Literal` cannot be cached as a unit.
    async fn lookup(&self, op: &str, args: &BTreeMap<String, Value>) -> Lookup;

    /// Insert a freshly-computed value for the given `(op, args)` pair.
    /// Implementations are free to evict existing entries.
    async fn insert(&self, op: &str, args: &BTreeMap<String, Value>, value: &Value);

    /// Convenience wrapper for callers that want the `Compose` form to be
    /// hashed by structure rather than by name. Default impl returns
    /// [`Lookup::Miss`]; Phase 3 may override.
    async fn lookup_fragment(&self, _fragment: &VaHera) -> Lookup {
        Lookup::Miss
    }
}

/// Phase-1 no-op CMM: every lookup misses; insertions are dropped.
#[derive(Default)]
pub struct NoopCmm;

#[async_trait::async_trait]
impl Subsystem for NoopCmm {
    fn name(&self) -> &'static str {
        "cmm"
    }
}

#[async_trait::async_trait]
impl Cmm for NoopCmm {
    async fn lookup(&self, _op: &str, _args: &BTreeMap<String, Value>) -> Lookup {
        Lookup::Miss
    }

    async fn insert(&self, _op: &str, _args: &BTreeMap<String, Value>, _value: &Value) {}
}
