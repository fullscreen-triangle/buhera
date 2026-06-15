//! Buhera kernel.
//!
//! The [`Kernel`] orchestrates five subsystems:
//!
//! * [`cmm::Cmm`] — Categorical Memory Manager (S-coord → address store).
//! * [`pss::Pss`] — Penultimate State Scheduler (heap by trajectory distance).
//! * [`dic::Dic`] — Demon I/O Controller (surgical retrieval, zero-cost sort).
//! * [`pve::Pve`] — Proof Validation Engine (vaHera statement-type checker).
//! * [`tem::Tem`] — Triple Equivalence Monitor (oscillator/cat/partition entropy).
//!
//! The crate is a faithful port of `driven/system/kernel/*.py`. It depends
//! only on [`buhera_substrate`] within the workspace; no external project
//! is touched.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cmm;
pub mod dic;
pub mod pss;
pub mod pve;
pub mod tem;
mod kernel;

pub use kernel::{Kernel, KernelError, KernelStats};
pub use cmm::{Cmm, MemoryObject, Tier};
pub use dic::{Dic, DicStats, RetrievedItem};
pub use pss::{Pss, Process, ProcessState};
pub use pve::{Pve, PveError, PveStats};
pub use tem::{Tem, TemSample, TemStats};

// Re-export the substrate types kernel consumers usually need so they
// can `use buhera_kernel::*` without depending on `buhera-substrate`
// directly when they don't have to.
pub use buhera_substrate::{SCoord, Trajectory};
