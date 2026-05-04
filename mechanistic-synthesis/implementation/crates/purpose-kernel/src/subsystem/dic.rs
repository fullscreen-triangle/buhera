//! DIC — Demon I/O Controller.
//!
//! Grounded in UTL §6 (Coupling estimation): given a query and a candidate
//! source, fetch only the bits whose mutual information with the query
//! exceeds a coupling-derived threshold $g^*$. DIC is opt-in per provider;
//! providers that have not been wrapped go through the standard executor
//! path unchanged.
//!
//! Phase 1 ships a no-op DIC that always recommends "fetch everything".
//! Phase 6 implements the mutual-information estimator (mirror of
//! `driven/src/utl/validate_14_coupling_estimator.py`) and the threshold
//! policy.

use crate::subsystem::Subsystem;

/// Recommendation from DIC about how aggressively to retrieve from a source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RetrievalPolicy {
    /// Fetch the entire source. Phase-1 default.
    Full,
    /// Fetch only the top-`k` highest-coupling fields/records. Phase 6
    /// chooses `k` from the live coupling-estimator.
    TopK(usize),
}

/// Pre-fetch hook consulted by retrieval-aware providers.
#[async_trait::async_trait]
pub trait Dic: Subsystem {
    /// Recommend a retrieval policy for the given `(query, source)` pair.
    /// Both arguments are opaque strings in Phase 1; Phase 6 introduces a
    /// typed query/source pair and a coupling estimate.
    async fn policy(&self, _query: &str, _source: &str) -> RetrievalPolicy {
        RetrievalPolicy::Full
    }
}

/// Phase-1 no-op DIC: always recommends [`RetrievalPolicy::Full`].
#[derive(Default)]
pub struct NoopDic;

#[async_trait::async_trait]
impl Subsystem for NoopDic {
    fn name(&self) -> &'static str {
        "dic"
    }
}

#[async_trait::async_trait]
impl Dic for NoopDic {}
