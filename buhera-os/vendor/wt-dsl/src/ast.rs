//! The `.wt` script AST.
//!
//! A script has four blocks, all optional except `scope`:
//!
//! ```text
//! scope <Name>:        what to analyse   — paths, language, repo
//! analyse:             which phases to run and with what parameters
//! assert:              the conditions that decide the verdict
//! report:              what to emit
//! ```
//!
//! The AST is deliberately close to the surface syntax: a directive that
//! parsed is a directive the user wrote. Nothing is inferred here, because a
//! script that silently means something other than what it says is worse than
//! one that fails to parse.

use serde::{Deserialize, Serialize};

// ── Script ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Script {
    pub scope:    Scope,
    pub analyse:  Analyse,
    pub asserts:  Vec<Assertion>,
    pub report:   Report,
}

// ── scope ─────────────────────────────────────────────────────────────────────

/// Which source files take part in the analysis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Scope {
    /// The `scope <Name>:` label. Cosmetic — it names the run in reports.
    pub name:     String,
    /// `repo "..."` — informational only; the CLI always analyses a local path.
    pub repo:     Option<String>,
    /// `include "glob"` — if empty, the whole project is in scope.
    pub include:  Vec<String>,
    /// `exclude "glob"` — applied after `include`.
    pub exclude:  Vec<String>,
    /// `language <ident>` — restricts extraction to one language.
    pub language: Option<String>,
}

// ── analyse ───────────────────────────────────────────────────────────────────

/// Which phases to run. `static` is implied whenever any phase is requested,
/// because the later phases consume its graph.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Analyse {
    pub static_phase: bool,
    /// `cycles through [...] max_depth N`
    pub cycles:       Option<Cycles>,
    /// `purpose ablate [...]`
    pub purpose:      Option<Purpose>,
    /// `dynamic traces "dir"`
    pub dynamic:      Option<Dynamic>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Cycles {
    /// Units the cycle must pass through. Empty means "all cycles".
    pub through:   Vec<String>,
    /// `max_depth N` — cap for Johnson's enumeration.
    pub max_depth: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Purpose {
    /// Units to ablate. Empty means "every unit", which is O(n) runs.
    pub ablate: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dynamic {
    pub traces: String,
}

// ── assert ────────────────────────────────────────────────────────────────────

/// One assertion. Each maps to exactly one check against `WindTunnelMetric`,
/// so a failure can always name the field that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Assertion {
    /// `regime >= Coherent`
    Regime { op: CmpOp, regime: String },
    /// `r_est >= 0.75`
    Scalar { field: ScalarField, op: CmpOp, value: f64 },
    /// `no holonomy_violations`
    NoHolonomyViolations,
    /// `no cycles`
    NoCycles,
    /// `purposeless none in ["a", "b"]` — none of these units may be
    /// purposeless. An empty list means *no unit anywhere* may be.
    PurposelessNoneIn { units: Vec<String> },
}

/// Numeric fields an assertion may compare against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarField {
    REst,
    RDyn,
    KC,
    SFlatEst,
}

impl ScalarField {
    pub fn as_str(&self) -> &'static str {
        match self {
            ScalarField::REst     => "r_est",
            ScalarField::RDyn     => "r_dyn",
            ScalarField::KC       => "k_c",
            ScalarField::SFlatEst => "s_flat_est",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "r_est"      => Some(ScalarField::REst),
            "r_dyn"      => Some(ScalarField::RDyn),
            "k_c"        => Some(ScalarField::KC),
            "s_flat_est" => Some(ScalarField::SFlatEst),
            _            => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CmpOp { Ge, Gt, Le, Lt, Eq }

impl CmpOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            CmpOp::Ge => ">=",
            CmpOp::Gt => ">",
            CmpOp::Le => "<=",
            CmpOp::Lt => "<",
            CmpOp::Eq => "==",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            ">=" => Some(CmpOp::Ge),
            ">"  => Some(CmpOp::Gt),
            "<=" => Some(CmpOp::Le),
            "<"  => Some(CmpOp::Lt),
            "==" => Some(CmpOp::Eq),
            _    => None,
        }
    }

    /// Compare two orderable values. Float equality uses a tolerance because
    /// exact `==` on a computed f64 is a bug waiting to be reported as a
    /// mysterious assertion failure.
    pub fn apply(&self, lhs: f64, rhs: f64) -> bool {
        match self {
            CmpOp::Ge => lhs >= rhs,
            CmpOp::Gt => lhs >  rhs,
            CmpOp::Le => lhs <= rhs,
            CmpOp::Lt => lhs <  rhs,
            CmpOp::Eq => (lhs - rhs).abs() < 1e-9,
        }
    }
}

// ── report ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub format:  Format,
    /// `include <section>` — empty means "everything".
    pub include: Vec<String>,
}

impl Default for Report {
    fn default() -> Self {
        Report { format: Format::Text, include: Vec::new() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format { Text, Json }
