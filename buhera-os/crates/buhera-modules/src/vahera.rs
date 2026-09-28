//! `vahera` — the Buhera OS's own language, executed on a module-owned
//! kernel (specification `specs/vahera.md`). Wraps `buhera-vahera` +
//! `buhera-kernel` exactly as the gateway's `/api/run` path does, rendering
//! results with the same shared `render_result`.

use buhera_kernel::Kernel;
use buhera_registry::{
    field_str, instruction_kind, ActResult, BindingKind, Descriptor, DslEntry, DslError, Instruction, Module,
    Validation,
};
use buhera_vahera::{execute_vahera, parse_vahera, render_result, MoleculeDatabase};
use serde_json::{json, Value};

/// Registry id.
pub const ID: &str = "vahera";

/// Ternary-address depth; matches the REPL and the gateway so a statement
/// resolves to the same address on every host.
pub const DEPTH: usize = 12;

/// Demo script (a subset of `examples/demo.bvh` that needs no molecule data).
pub const DEMO: &str = r#"memory store "groceries" = "milk, eggs, bread and coffee"
memory store "travel" = "flight to Munich on Friday morning"
memory find nearest "shopping list" k=1
demon sort
controller verify
kernel stats"#;

/// The module; its kernel persists between acts (contract M2).
pub struct Vahera {
    kernel: Kernel,
    molecules: MoleculeDatabase,
}

impl std::fmt::Debug for Vahera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vahera").finish_non_exhaustive()
    }
}

impl Default for Vahera {
    fn default() -> Self {
        Self::new()
    }
}

impl Vahera {
    /// Fresh kernel.
    pub fn new() -> Self {
        Self { kernel: Kernel::new(DEPTH), molecules: MoleculeDatabase::new() }
    }
}

/// The vaHera front end (`buhera_vahera::parse_vahera`).
pub fn validate(source: &str) -> Validation {
    match parse_vahera(source) {
        Ok(_) => Validation::valid(),
        Err(e) => {
            let msg = e.to_string();
            Validation::invalid(vec![match buhera_registry::line_from_message(&msg) {
                Some(l) => DslError::at(msg, l, None),
                None => DslError::msg(msg),
            }])
        }
    }
}

/// DSL registry entry.
pub fn dsl() -> DslEntry {
    DslEntry { id: "vahera", label: "vaHera", extension: ".vhr", module_id: ID, pack_id: "vahera", validate }
}

impl Module for Vahera {
    fn id(&self) -> &str {
        ID
    }

    fn describe(&self) -> Descriptor {
        Descriptor {
            id: ID.into(),
            description: "vaHera — the Buhera OS language: describe/resolve/spawn targets, navigate to the \
                          penultimate state, store and retrieve content at categorical addresses, zero-cost sort, \
                          triple-equivalence verification. Runs on a kernel owned by this module."
                .into(),
            instructions: vec![
                r#"dispatch("vahera", "demo")"#.into(),
                r#"dispatch("vahera", "memory store \"a\" = \"hello\"\nmemory list")"#.into(),
                r#"dispatch("vahera", { kind: "reset" })"#.into(),
            ],
            dsl: Some("vahera".into()),
            binding: BindingKind::Native,
        }
    }

    fn execute(&mut self, instruction: &Instruction, _act_budget: u32) -> ActResult {
        let source = match (instruction_kind(instruction), instruction.is_object()) {
            (Some("demo"), false) => DEMO.to_string(),
            (Some(s), false) => s.to_string(),
            (Some("reset"), true) => {
                *self = Self::new();
                return ActResult::text(&["vahera: kernel reset"], 0.0);
            }
            (Some("run"), true) => match field_str(instruction, "source") {
                Some(s) => s.to_string(),
                None => return ActResult::invalid(ID, "{ kind: \"run\", source: string }"),
            },
            _ => return ActResult::invalid(ID, "vaHera source, \"demo\", { kind: \"run\", source } or { kind: \"reset\" }"),
        };
        match execute_vahera(&source, &mut self.kernel, &self.molecules) {
            Ok(ctx) => {
                let results: Vec<Value> = ctx.results.iter().map(render_result).collect();
                let n = results.len();
                ActResult::done(
                    json!({ "kind": "vahera_result", "summary": format!("vahera: {n} result(s)"),
                            "results": results, "trace": ctx.trace }),
                    // A count of what the act produced: rendered results. vaHera has
                    // no distance-to-solution notion (contract A3).
                    n as f64,
                )
            }
            Err(e) => ActResult::fail(&[&format!("vahera: {e}")], e.to_string()),
        }
    }
}
