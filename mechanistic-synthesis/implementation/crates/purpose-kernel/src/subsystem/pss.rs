//! PSS — Penultimate State Scheduler.
//!
//! Grounded in UTL Theorem 4 (Critical Slowing): near regime boundaries
//! $R_b \in \{0.3, 0.5, 0.8, 0.95\}$ the relaxation time diverges as
//! $\tau_\mathrm{relax} \propto |R - R_b|^{-1}$. PSS uses the live $R$
//! estimate to order pending dispatches by predicted slack.
//!
//! Phase 1 ships a no-op scheduler that preserves arrival order. Phase 4
//! implements the online $R$ estimator (mirror of `compute_R` in
//! `driven/src/utl/validate_06_phase_coherence.py`) and the boundary-aware
//! ordering policy.

use purpose_core::VaHera;

use crate::subsystem::Subsystem;

/// Decision returned by [`Pss::order`] for a single pending fragment.
///
/// In Phase 1 the kernel never queues fragments — every dispatch runs as
/// soon as it arrives — so this enum is unused at runtime, but it pins the
/// trait surface that Phase 4 will occupy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Run the fragment immediately.
    Now,
    /// Defer the fragment by `n` ticks; the kernel re-asks PSS each tick.
    Defer(u32),
}

/// Pre-dispatch ordering hook.
#[async_trait::async_trait]
pub trait Pss: Subsystem {
    /// Decide whether the given fragment should run immediately or be
    /// deferred. Default impl returns [`Order::Now`].
    async fn order(&self, _fragment: &VaHera) -> Order {
        Order::Now
    }

    /// Notify PSS that a dispatch has just completed. Phase 4 uses this to
    /// update the online $R$ estimate from the post-dispatch class
    /// assignment. Default impl is a no-op.
    async fn observe_completion(&self, _fragment: &VaHera) {}
}

/// Phase-1 no-op PSS: every fragment runs immediately, in arrival order.
#[derive(Default)]
pub struct NoopPss;

#[async_trait::async_trait]
impl Subsystem for NoopPss {
    fn name(&self) -> &'static str {
        "pss"
    }
}

#[async_trait::async_trait]
impl Pss for NoopPss {}
