//! TEM — Triple Equivalence Monitor.
//!
//! Grounded in COE Theorem 1 (Time-Count Identity, $t = M/f$), Theorem 7
//! (MTIC: $Q$, $t$, $M$, identity are unit conversions of one quantity),
//! and Theorem 8 (Sliding-Endpoint: reproducibility ⇔ irreversibility of
//! $M$). TEM samples the kernel's dispatch trace at a configurable cadence
//! and alarms on any violation of these conservation laws.
//!
//! Phase 1 ships a no-op TEM that ignores every event. Phase 5 implements
//! the four invariants as independent samplers running on a `tokio::spawn`ed
//! task.

use crate::event::DispatchEvent;
use crate::subsystem::Subsystem;

/// Subsystem that observes the dispatch event stream and reports invariant
/// violations.
#[async_trait::async_trait]
pub trait Tem: Subsystem {
    /// Inspect a single dispatch event. Default impl is a no-op; Phase 5
    /// implementations check monotone-$M$, the Time-Count Identity, the
    /// Reproducibility property, and the Sliding-Endpoint Theorem.
    async fn observe(&self, _event: &DispatchEvent) {}
}

/// Phase-1 no-op TEM: ignores every event.
#[derive(Default)]
pub struct NoopTem;

#[async_trait::async_trait]
impl Subsystem for NoopTem {
    fn name(&self) -> &'static str {
        "tem"
    }
}

#[async_trait::async_trait]
impl Tem for NoopTem {}
