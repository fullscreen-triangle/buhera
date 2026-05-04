//! Five kernel subsystems and their common super-trait.
//!
//! Each subsystem lives in its own submodule with a single trait; concrete
//! implementations and Phase-1 no-op stubs live next to the trait. Phases 2–6
//! of the implementation plan (`long-grass/implementation-plan.md`) replace
//! the no-op stubs with the real implementations one at a time.

pub mod cmm;
pub mod dic;
pub mod pss;
pub mod pve;
pub mod tem;

/// Super-trait shared by every kernel subsystem.
///
/// `name()` is used by the kernel for telemetry and structured logs. `tick()`
/// is invoked by the kernel between dispatches once per dispatch (default
/// implementation does nothing); subsystems may override to do bookkeeping
/// that does not need to run inline with the dispatch path.
#[async_trait::async_trait]
pub trait Subsystem: Send + Sync {
    /// Stable identifier for telemetry. Should be a short kebab-case string
    /// (e.g. `"cmm"`, `"pve"`).
    fn name(&self) -> &'static str;

    /// Called once per dispatch after the executor has returned. The kernel
    /// awaits each tick in turn; ticks must therefore be cheap. Default
    /// implementation is a no-op.
    async fn tick(&self) {}
}
