//! Vendored χ from `bloodhound/thrust/tracker` (local shim; `chi.rs` is
//! upstream verbatim). Upstream is a binary crate with private modules, so
//! this crate re-roots them as a library: `chi` is upstream's module, and
//! `purpose` provides the two index types it reads.
#![allow(dead_code, missing_docs)]

pub mod chi;
pub mod purpose;
