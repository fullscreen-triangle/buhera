//! PVE — Proof Validation Engine.
//!
//! Grounded in COE Theorem 5 (Three-Route Equivalence): every operation's
//! computational weight $Q$ admits three independent measurements (Residue,
//! Confinement, Negation Fixed Point) that must agree. PVE rejects any
//! fragment whose three-route weights diverge.
//!
//! Phase 1 ships a no-op gate that admits every fragment. Phase 2 (see
//! `long-grass/implementation-plan.md`) implements the three-route audit
//! against the validators in `driven/src/coe/validate_03/04/05_route_*.py`.

use purpose_core::{Error, VaHera};

use crate::subsystem::Subsystem;

/// Pre-dispatch gate. Receives the fragment about to be executed and either
/// admits it or returns an error explaining the rejection.
///
/// Called inline with the dispatch path; the implementation must be cheap on
/// release builds.
#[async_trait::async_trait]
pub trait Pve: Subsystem {
    /// Validate the fragment. Phase 2 will additionally cross-check that
    /// `typecheck`'s output is consistent with the three-route weight estimate.
    async fn validate(&self, fragment: &VaHera) -> Result<(), Error>;
}

/// Phase-1 no-op PVE: admits every fragment.
///
/// Replaced in Phase 2 by `purpose_kernel::subsystem::pve::ThreeRoutePve`.
#[derive(Default)]
pub struct NoopPve;

#[async_trait::async_trait]
impl Subsystem for NoopPve {
    fn name(&self) -> &'static str {
        "pve"
    }
}

#[async_trait::async_trait]
impl Pve for NoopPve {
    async fn validate(&self, _fragment: &VaHera) -> Result<(), Error> {
        Ok(())
    }
}
