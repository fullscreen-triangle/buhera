//! Buhera federation registry.
//!
//! Three things, all engine-free:
//!
//! * [`contract`] — the [`Module`] trait and the JSON-shaped values that
//!   cross it ([`Instruction`], [`ActResult`], [`Descriptor`]).
//! * [`registry`] — [`Registry`]: register, dispatch, audit log,
//!   post-dispatch hooks.
//! * [`dsl`] — [`DslRegistry`]: language id → real validator + executing
//!   module + grounding pack.
//! * [`catalogue`] — the normative member list and the conformance check
//!   both registry libraries (this crate and `@buhera/registry`) pass.
//!
//! Engine adapters live in `buhera-modules`, one cargo feature per module,
//! so this crate never pulls an engine's dependency tree.
//!
//! Normative text: `specifications/architecture/02-module-contract.md`
//! through `05-catalogue.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod catalogue;
pub mod contract;
pub mod dsl;
pub mod registry;

pub use catalogue::{conformance, Catalogue, Host};
pub use contract::{
    field_str, instruction_kind, ActResult, BindingKind, Descriptor, Instruction, Module, OutputCell,
};
pub use dsl::{line_from_message, DslEntry, DslError, DslRegistry, DslRegistryError, DslSummary, Validation};
pub use registry::{AuditEntry, DispatchError, HookId, Registry};
