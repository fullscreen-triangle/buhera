//! Frozen type surface of the Purpose framework.
//!
//! Every symbol re-exported at the crate root is stability-contract
//! frozen per `long-grass/integration.md` §2.1. Additions allowed;
//! renames and removals are major-version events.

pub mod vahera;
pub mod value;
pub mod types;
pub mod operation;
pub mod domain;
pub mod resolver;
pub mod typecheck;
pub mod error;

pub use vahera::VaHera;
pub use value::Value;
pub use types::Type;
pub use operation::Operation;
pub use domain::Domain;
pub use resolver::Resolver;
pub use typecheck::typecheck;
pub use error::Error;
