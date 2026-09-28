//! The Module contract (specification §M, `specifications/architecture/02-module-contract.md`).
//!
//! A module is anything that can take an *instruction* and perform one *act*
//! against it. The contract is deliberately the same shape in Rust and in
//! TypeScript (`@buhera/registry`), so an act dispatched on either side is
//! recorded, audited and rendered identically, and so the TypeScript side can
//! forward an act to a Rust-hosted module over the wire without translation.
//!
//! Everything that crosses the contract is JSON-shaped (`serde_json::Value`):
//! instructions come from terminals, tutorials, generated DSL scripts and
//! remote callers, and must survive a round trip through JSON unchanged.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// What a caller asks a module to do.
///
/// By convention it is either a string (DSL source text, or a bare verb such
/// as `"demo"`) or an object carrying a `kind` field that names the
/// operation (`{ "kind": "run", "source": "…" }`). Each module's
/// specification enumerates the shapes it accepts; anything else must be
/// answered with an `invalid instruction` result, never a panic.
pub type Instruction = Value;

/// The outcome of one act.
///
/// Mirrors the TypeScript `ActResult` field for field. `output_delta` is the
/// renderable payload; by convention it is an object whose `kind` names the
/// renderer (`"text"` with `lines`, `"sbs_result"`, …).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActResult {
    /// Did the act achieve what the instruction asked for?
    pub ok: bool,
    /// What the act produced, for rendering. `None` only when a module
    /// failed so badly it produced nothing (the registry's panic path).
    pub output_delta: Option<Value>,
    /// How much work remains, in the module's own declared unit. Every
    /// module specification states what its residue counts; `0` means
    /// "nothing left to do", never "unknown".
    pub residue: f64,
    /// Did the act run to completion within its act budget? A module that
    /// honours budgets returns `false` when it stopped early and can be
    /// resumed.
    pub completed: bool,
    /// Machine-readable failure reason when `ok` is false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ActResult {
    /// A successful, completed act.
    pub fn done(output_delta: Value, residue: f64) -> Self {
        Self { ok: true, output_delta: Some(output_delta), residue, completed: true, error: None }
    }

    /// A failed, completed act rendered as text lines — the convention every
    /// module uses for user-facing errors.
    pub fn fail(lines: &[&str], error: impl Into<String>) -> Self {
        Self {
            ok: false,
            output_delta: Some(json!({ "kind": "text", "lines": lines })),
            residue: 0.0,
            completed: true,
            error: Some(error.into()),
        }
    }

    /// The standard answer to an instruction the module does not accept.
    pub fn invalid(module_id: &str, expected: &str) -> Self {
        Self::fail(&[&format!("{module_id}: instruction must be {expected}")], "invalid instruction")
    }

    /// A text-only successful result.
    pub fn text(lines: &[&str], residue: f64) -> Self {
        Self::done(json!({ "kind": "text", "lines": lines }), residue)
    }

    /// The `kind` field of the output delta, if there is one.
    pub fn delta_kind(&self) -> Option<&str> {
        self.output_delta.as_ref()?.get("kind")?.as_str()
    }
}

/// What kind of cell a host should allocate to render this module's
/// output (used by sufficiency checks before dispatch).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputCell {
    /// Renderer kind, conventionally `<module>_cell`.
    pub kind: String,
}

/// How a module reaches the engine that does the work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingKind {
    /// The engine is linked into this process.
    Native,
    /// The act is forwarded to another host's registry (TS → gateway).
    Remote,
    /// The act is executed by spawning a CLI and parsing its JSON.
    Bridge,
}

/// A module's self-description, returned by [`Module::describe`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Descriptor {
    /// Module id; equal to [`Module::id`].
    pub id: String,
    /// One-paragraph description for listings.
    pub description: String,
    /// Example invocations, as a user would type them.
    pub instructions: Vec<String>,
    /// The DSL this module executes, if it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dsl: Option<String>,
    /// How this build reaches the engine.
    pub binding: BindingKind,
}

/// A federation member.
///
/// `execute` takes `&mut self`: a module that keeps state between acts (a
/// kernel, a tracker's repo set) keeps it in the module value, in the open,
/// where the registry owns it — never in a global.
pub trait Module: Send {
    /// Stable identifier; the key the module is dispatched by.
    fn id(&self) -> &str;

    /// Self-description for listings and the catalogue conformance check.
    fn describe(&self) -> Descriptor;

    /// Perform one act. Must not panic on malformed input — return
    /// [`ActResult::invalid`] instead. (The registry does contain panics,
    /// but a contained panic is recorded as a defect.)
    fn execute(&mut self, instruction: &Instruction, act_budget: u32) -> ActResult;

    /// The cell this instruction's output needs.
    fn output_cell(&self, _instruction: &Instruction) -> OutputCell {
        OutputCell { kind: format!("{}_cell", self.id()) }
    }
}

/// Read the conventional `kind` of an instruction: the string itself for
/// a bare string, the `kind` field for an object, `None` otherwise.
pub fn instruction_kind(instruction: &Instruction) -> Option<&str> {
    match instruction {
        Value::String(s) => Some(s.as_str()),
        Value::Object(o) => o.get("kind").and_then(Value::as_str),
        _ => None,
    }
}

/// Fetch a string field from an object instruction.
pub fn field_str<'a>(instruction: &'a Instruction, key: &str) -> Option<&'a str> {
    instruction.get(key).and_then(Value::as_str)
}
