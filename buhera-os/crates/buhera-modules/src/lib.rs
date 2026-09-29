//! Buhera federation modules.
//!
//! Each module is a thin adapter from a vendored engine to the
//! [`buhera_registry::Module`] contract, behind a cargo feature of the same
//! name. [`federation`] builds the registry for whichever features are
//! enabled; with `full` it is the complete Rust-side federation and must pass
//! catalogue conformance (`tests/conformance.rs`).
//!
//! Normative text: `specifications/specs/<module>.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use buhera_registry::{DslRegistry, Registry};

#[cfg(feature = "heihachi")]
pub mod heihachi;
#[cfg(feature = "levinthal")]
pub mod levinthal;
#[cfg(feature = "mekaneck")]
pub mod mekaneck;
#[cfg(feature = "ndombolo")]
pub mod ndombolo;
#[cfg(feature = "olduvai")]
pub mod olduvai;
#[cfg(feature = "sbs-core")]
pub mod sbs_core;
#[cfg(feature = "tracker")]
pub mod tracker;
#[cfg(feature = "vahera")]
pub mod vahera;
#[cfg(feature = "windtunnel")]
pub mod windtunnel;
#[cfg(feature = "zangalewa")]
pub mod zangalewa;

/// Host capabilities that change which operations a module may perform.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// May modules read the local filesystem (tracker `character_at`/`list`)?
    /// False in the wasm build.
    pub filesystem: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { filesystem: true }
    }
}

/// Build the module registry and DSL registry for every enabled module.
pub fn federation(options: Options) -> (Registry, DslRegistry) {
    let mut modules = Registry::new();
    let mut dsls = DslRegistry::new();
    let _ = (&mut modules, &mut dsls, options);

    #[cfg(feature = "vahera")]
    {
        modules.register(Box::new(vahera::Vahera::new()));
        dsls.register(vahera::dsl());
    }
    #[cfg(feature = "ndombolo")]
    {
        modules.register(Box::new(ndombolo::Ndombolo::new()));
        dsls.register(ndombolo::dsl());
    }
    #[cfg(feature = "windtunnel")]
    {
        modules.register(Box::new(windtunnel::WindTunnel::new()));
        dsls.register(windtunnel::dsl());
    }
    #[cfg(feature = "tracker")]
    {
        modules.register(Box::new(tracker::Tracker { filesystem: options.filesystem }));
    }
    #[cfg(feature = "heihachi")]
    {
        modules.register(Box::new(heihachi::Heihachi::new()));
        for d in heihachi::dsls() {
            dsls.register(d);
        }
    }
    #[cfg(feature = "levinthal")]
    {
        modules.register(Box::new(levinthal::Levinthal::new()));
    }
    #[cfg(feature = "mekaneck")]
    {
        modules.register(Box::new(mekaneck::Mekaneck::new()));
        dsls.register(mekaneck::dsl());
    }
    #[cfg(feature = "olduvai")]
    {
        modules.register(Box::new(olduvai::Olduvai::new()));
    }
    #[cfg(feature = "sbs-core")]
    {
        modules.register(Box::new(sbs_core::SbsCore::new()));
    }
    #[cfg(feature = "zangalewa")]
    {
        modules.register(Box::new(zangalewa::Zangalewa::new()));
    }
    (modules, dsls)
}
