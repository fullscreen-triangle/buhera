//! Buhera categorical substrate.
//!
//! Pure mathematics that every subsystem consumes:
//!
//! * [`SCoord`] — points in S-entropy space `[0,1]^3`.
//! * [`embed_text`], [`embed_molecule`] — deterministic content → `SCoord`.
//! * [`fisher_distance_1d`], [`s_distance`] — Fisher information metric.
//! * [`ternary_address`], [`common_prefix_length`] — categorical addresses.
//! * [`backward_navigate`], [`completion_morphism`], [`Trajectory`] —
//!   the backward trajectory completion algorithm whose termination at
//!   the penultimate state is the structural requirement for recognition.
//! * [`nearest`] — k-nearest neighbour query under `s_distance`.
//!
//! This crate has no runtime dependencies beyond `serde`. It is a
//! faithful port of `driven/system/substrate.py`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod scoord;
mod embed;
mod fisher;
mod ternary;
mod trajectory;

pub use scoord::{SCoord, SCoordError};
pub use embed::{embed_text, embed_molecule, MoleculeProperties};
pub use fisher::{fisher_distance_1d, s_distance};
pub use ternary::{ternary_address, common_prefix_length};
pub use trajectory::{Trajectory, backward_navigate, completion_morphism, nearest};
