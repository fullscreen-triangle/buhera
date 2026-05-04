//! Typed kernel events.
//!
//! Subsystems publish events through the kernel's [`Observable`] handles
//! (see [`crate::observable`]). External consumers — TEM, log exporters,
//! tracing layers — subscribe without coupling to internal subsystem state.
//!
//! [`Observable`]: crate::observable::Observable

use std::time::Duration;

/// One end-to-end dispatch through the kernel.
///
/// Emitted by [`crate::kernel::BuheraKernel::dispatch`] after the executor
/// returns (whether successfully or with an error). All later phases of the
/// implementation plan rely on this event being present and well-formed:
///
/// * Phase 3 (CMM) populates `cache_hit` from the categorical-memory-manager
///   path that runs before dispatch.
/// * Phase 4 (PSS) increments `decisions_consumed` in step with the kernel's
///   internal $M$ count, which is the operational definition of kernel time
///   (COE Theorem 1, Time-Count Identity).
/// * Phase 5 (TEM) treats this event stream as the dispatch trace whose
///   conservation invariants it samples.
#[derive(Debug, Clone)]
pub struct DispatchEvent {
    /// The top-level operation name — `None` for a [`Compose`] root or a
    /// [`Literal`].
    ///
    /// [`Compose`]: purpose_core::VaHera::Compose
    /// [`Literal`]: purpose_core::VaHera::Literal
    pub op: Option<String>,
    /// Wall-clock duration of the dispatch.
    pub elapsed: Duration,
    /// Whether the dispatch returned [`Ok`].
    pub ok: bool,
    /// `true` once the CMM is implemented and reported a cache hit; always
    /// `false` in Phase 1.
    pub cache_hit: bool,
    /// Increment to the kernel's monotone decision count $M$. Always `1` for
    /// a single dispatch in Phase 1; later phases account for sub-dispatches
    /// (Compose nodes, three-route audit decisions in PVE).
    pub decisions_consumed: u64,
}

impl DispatchEvent {
    /// Construct an event whose CMM and route-audit fields are still
    /// Phase-1 defaults. Called from [`crate::kernel::BuheraKernel::dispatch`].
    pub fn new(op: Option<String>, elapsed: Duration, ok: bool) -> Self {
        Self {
            op,
            elapsed,
            ok,
            cache_hit: false,
            decisions_consumed: 1,
        }
    }
}
