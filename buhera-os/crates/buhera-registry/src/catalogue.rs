//! The catalogue (specification `05-catalogue.md`).
//!
//! `specifications/registry/catalogue.json` is the single normative list of
//! federation members: which modules exist, which languages they own, where
//! their engines come from, and how each language binding reaches them.
//! Both registry libraries are checked against it by [`conformance`]; the
//! documentation site draws its diagrams from it.
//!
//! The library build never reads the file (the OS stays standalone); tests
//! and hosts load it and hand it in.

use serde::{Deserialize, Serialize};

use crate::contract::BindingKind;
use crate::dsl::DslRegistry;
use crate::registry::Registry;

/// Schema tag this crate understands.
pub const SCHEMA: &str = "buhera.catalogue/1";

/// Which host language a binding column describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Host {
    /// `buhera-registry` + `buhera-modules`.
    Rust,
    /// `@buhera/registry`.
    Ts,
}

/// A binding column entry: how a host reaches the module, or `none`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Binding {
    /// Engine linked in-process.
    Native,
    /// Forwarded to another host's registry.
    Remote,
    /// Spawned CLI.
    Bridge,
    /// Not available on this host.
    None,
}

impl Binding {
    /// The contract-level kind, or `None` for [`Binding::None`].
    pub fn kind(self) -> Option<BindingKind> {
        match self {
            Binding::Native => Some(BindingKind::Native),
            Binding::Remote => Some(BindingKind::Remote),
            Binding::Bridge => Some(BindingKind::Bridge),
            Binding::None => None,
        }
    }
}

/// Per-host bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bindings {
    /// Rust host.
    pub rust: Binding,
    /// TypeScript host.
    pub ts: Binding,
}

impl Bindings {
    /// The binding for a host.
    pub fn for_host(&self, host: Host) -> Binding {
        match host {
            Host::Rust => self.rust,
            Host::Ts => self.ts,
        }
    }
}

/// Where an engine's source of truth lives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Upstream {
    /// `owner/repo` on GitHub.
    pub repo: String,
    /// Path inside the repo.
    pub path: String,
    /// `rust`, `ts` or `js`.
    pub language: String,
    /// The commit the vendored copy was taken from.
    pub commit: String,
    /// Where the vendored copy lives in this repo, if vendored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendored_at: Option<String>,
}

/// One module row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleRow {
    /// Registry id.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Architectural layer (`language`, `science`, `runtime`, `observation`,
    /// `coordination`).
    pub layer: String,
    /// One-sentence summary.
    pub summary: String,
    /// Language id this module executes, if any.
    #[serde(default)]
    pub dsl: Option<String>,
    /// Engine sources.
    #[serde(default)]
    pub upstream: Vec<Upstream>,
    /// How each host reaches it.
    pub bindings: Bindings,
    /// `output_delta.kind` values a successful act can carry.
    #[serde(default)]
    pub output_kinds: Vec<String>,
    /// What `residue` counts for this module.
    pub residue: String,
    /// Declared side effects (network, filesystem, processes, clock, GPU).
    #[serde(default)]
    pub side_effects: Vec<String>,
    /// Path of the module's specification, relative to `specifications/`.
    pub spec: String,
}

/// One language row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DslRow {
    /// Language id.
    pub id: String,
    /// Display name.
    pub label: String,
    /// File extension with dot.
    pub extension: String,
    /// Executing module id.
    pub module_id: String,
    /// Grounding pack id.
    pub pack_id: String,
    /// Hosts whose DSL registry carries a validator for it.
    pub validators: Vec<Host>,
}

/// The whole catalogue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Catalogue {
    /// Must equal [`SCHEMA`].
    pub schema: String,
    /// Modules.
    pub modules: Vec<ModuleRow>,
    /// Languages.
    pub dsls: Vec<DslRow>,
}

impl Catalogue {
    /// Parse from JSON text.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Look up a module row.
    pub fn module(&self, id: &str) -> Option<&ModuleRow> {
        self.modules.iter().find(|m| m.id == id)
    }
}

/// Check a host's registries against the catalogue. Returns every
/// violation found; empty means conformant.
///
/// Rules (spec `05-catalogue.md` §3):
/// * C1 — every module whose binding for `host` is not `none` is registered,
///   and its descriptor's binding kind equals the catalogue's.
/// * C2 — every registered module appears in the catalogue with a non-`none`
///   binding for `host` (no unlisted members).
/// * C3 — a registered module that declares a DSL declares the catalogue's.
/// * C4 — every language listing `host` among its validators is registered
///   in the DSL registry, routed to the catalogue's module and pack.
/// * C5 — every registered language appears in the catalogue for `host`.
pub fn conformance(catalogue: &Catalogue, host: Host, modules: &Registry, dsls: &DslRegistry) -> Vec<String> {
    let mut v = Vec::new();
    if catalogue.schema != SCHEMA {
        v.push(format!("schema: expected {SCHEMA}, found {}", catalogue.schema));
    }
    let descriptors = modules.list();
    for row in &catalogue.modules {
        let want = row.bindings.for_host(host);
        let got = descriptors.iter().find(|d| d.id == row.id);
        match (want.kind(), got) {
            (Some(kind), Some(d)) => {
                if d.binding != kind {
                    v.push(format!("C1 {}: binding {:?} ≠ catalogue {:?}", row.id, d.binding, kind));
                }
                if d.dsl.is_some() && d.dsl != row.dsl {
                    v.push(format!("C3 {}: dsl {:?} ≠ catalogue {:?}", row.id, d.dsl, row.dsl));
                }
            }
            (Some(_), None) => v.push(format!("C1 {}: listed for {host:?} but not registered", row.id)),
            (None, Some(_)) => v.push(format!("C2 {}: registered but catalogue says none for {host:?}", row.id)),
            (None, None) => {}
        }
    }
    for d in &descriptors {
        if catalogue.module(&d.id).is_none() {
            v.push(format!("C2 {}: registered but not in the catalogue", d.id));
        }
    }
    for row in &catalogue.dsls {
        let listed = row.validators.contains(&host);
        match (listed, dsls.get(&row.id)) {
            (true, Some(e)) => {
                if e.module_id != row.module_id || e.pack_id != row.pack_id {
                    v.push(format!(
                        "C4 {}: routes to {}/{} ≠ catalogue {}/{}",
                        row.id, e.module_id, e.pack_id, row.module_id, row.pack_id
                    ));
                }
            }
            (true, None) => v.push(format!("C4 {}: validator listed for {host:?} but not registered", row.id)),
            (false, Some(_)) => v.push(format!("C5 {}: registered but catalogue lists no {host:?} validator", row.id)),
            (false, None) => {}
        }
    }
    for id in dsls.ids() {
        if !catalogue.dsls.iter().any(|r| r.id == id) {
            v.push(format!("C5 {id}: registered language not in the catalogue"));
        }
    }
    v
}
