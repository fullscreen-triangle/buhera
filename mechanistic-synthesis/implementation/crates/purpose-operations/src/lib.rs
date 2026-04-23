//! Execution layer of the Purpose framework.
//!
//! Frozen surface per integration.md §2.1:
//! * `Provider` trait — permanent signature
//! * `OperationRegistry::register` — stable API
//!
//! Additions allowed; renames and removals are major-version events.

pub mod provider;
pub mod registry;
pub mod executor;
pub mod providers;

pub use provider::Provider;
pub use registry::OperationRegistry;
pub use executor::Executor;
