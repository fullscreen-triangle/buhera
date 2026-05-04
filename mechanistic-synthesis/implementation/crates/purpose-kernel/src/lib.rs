//! Buhera Kernel — Phase 1 skeleton.
//!
//! Wraps [`purpose_operations::Executor`] with five subsystems whose contracts
//! are formally grounded in the two papers under `long-grass/docs/`:
//!
//! * **PVE** Proof Validation Engine — COE Theorem 5 (Three-Route Equivalence).
//! * **CMM** Categorical Memory Manager — UTL Theorem 5 (Cache Extinction).
//! * **PSS** Penultimate State Scheduler — UTL Theorem 4 (Critical Slowing).
//! * **DIC** Demon I/O Controller — UTL §6 (Coupling estimation).
//! * **TEM** Triple Equivalence Monitor — COE Theorem 7 (MTIC), Theorem 8
//!   (Sliding-Endpoint), and the monotone-log corollary.
//!
//! In Phase 1 every subsystem is a no-op: dispatch through the kernel produces
//! identical results to a direct call on the bare `Executor`. Subsequent phases
//! (see `long-grass/implementation-plan.md`) replace each no-op with the real
//! implementation; the trait surface defined here is permanent.
//!
//! # Stability
//!
//! The frozen interfaces in `integration.md` §2.1 (`VaHera`, `Value`, `Type`,
//! `Operation`, `Domain`, `Resolver`, `Provider`, `OperationRegistry::register`)
//! are not modified. The kernel reads vaHera fragments and dispatches their
//! `Call` nodes through the same provider pathway as the bare executor; it
//! never rewrites a fragment.
//!
//! New stability contract introduced by this crate:
//!
//! * [`Subsystem`] — single super-trait with a `name()` and an async `tick()`.
//! * [`Cmm`], [`Pss`], [`Dic`], [`Pve`], [`Tem`] — one trait per kernel
//!   responsibility. Methods may be added with default impls; renames and
//!   removals are major-version events.
//! * [`Observable`] — read-only handle over a broadcast channel of typed
//!   kernel events.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod event;
pub mod observable;
pub mod subsystem;
pub mod kernel;

pub use event::DispatchEvent;
pub use observable::Observable;
pub use subsystem::{
    Subsystem,
    cmm::{Cmm, NoopCmm},
    dic::{Dic, NoopDic},
    pss::{Pss, NoopPss},
    pve::{Pve, NoopPve},
    tem::{NoopTem, Tem},
};
pub use kernel::{BuheraKernel, BuheraKernelBuilder};
