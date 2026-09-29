//! `mekaneck` — the Mekaneck (`.mck`) inquiry language (specification
//! `specs/mekaneck.md`). Wraps vendored `mekaneck-lang` (lexer, LL(1)
//! parser, type checker, small-step evaluator) over `mekaneck-algebra`.
//!
//! One primitive, `seek`: a positive description plus a mandatory exclusion,
//! supported by at least three mutually independent catalysts, terminating
//! by closure. A seek is `Resolved { cell }` or `Declined { cells }` — and a
//! declination is a result, not a failure.
//!
//! `run` mirrors upstream's CLI (`crates/cli/src/main.rs`, `cmd_run`) and
//! nothing more: parse → typecheck → a fixed substrate of caller-supplied
//! cells → `eval_seek` for every `let`. There is no substrate in-host: the
//! cells a catalyst reads are the caller's to state.

use std::collections::BTreeMap;

use buhera_registry::{
    field_str, ActResult, BindingKind, Descriptor, DslEntry, DslError, Instruction, Module, Validation,
};
use mekaneck_lang::{self as lang, FloorValues, Severity, Span};
use serde_json::{json, Value};

/// Registry id.
pub const ID: &str = "mekaneck";

/// Upstream `chatelier/examples/coherence.mck`.
pub const DEMO_SOURCE: &str = include_str!("../../../vendor/mekaneck/examples/coherence.mck");
/// The README's run of the demo: every catalyst reads the same cell.
pub const DEMO_CELLS: [(&str, &str); 3] = [("spectral", "high"), ("surrogate", "high"), ("phase", "high")];

fn line_col(src: &str, span: Span) -> (u32, u32) {
    let (l, c) = span.line_col(src);
    (l as u32, c as u32)
}

fn diagnostics_json(src: &str, diags: &[lang::Diagnostic]) -> Vec<Value> {
    diags
        .iter()
        .map(|d| {
            let (line, column) = line_col(src, d.span);
            json!({
                "severity": if d.severity == Severity::Error { "error" } else { "warning" },
                "message": d.message, "line": line, "column": column,
            })
        })
        .collect()
}

/// The Mekaneck front end with no substrate floors supplied (an unsupplied
/// floor is a warning, not an error).
pub fn validate(source: &str) -> Validation {
    let errors: Vec<DslError> = lang::diagnose(source, &FloorValues::new())
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| {
            let (line, column) = line_col(source, d.span);
            DslError::at(d.message, line, Some(column))
        })
        .collect();
    if errors.is_empty() {
        Validation::valid()
    } else {
        Validation::invalid(errors)
    }
}

/// DSL registry entry.
pub fn dsl() -> DslEntry {
    DslEntry { id: "mekaneck", label: "Mekaneck", extension: ".mck", module_id: ID, pack_id: "mekaneck", validate }
}

fn floors_of(instruction: &Instruction) -> Result<FloorValues, String> {
    let mut out = BTreeMap::new();
    if let Some(v) = instruction.get("floors") {
        let obj = v.as_object().ok_or("floors must be { substrate: number }")?;
        for (k, x) in obj {
            out.insert(k.clone(), x.as_f64().ok_or_else(|| format!("floor for {k:?} must be a number"))?);
        }
    }
    Ok(out)
}

fn refused(source: &str, err: &lang::Error) -> ActResult {
    let (line, column) = err.span().map(|s| line_col(source, s)).unwrap_or((0, 0));
    let message = err.to_string();
    ActResult {
        ok: false,
        output_delta: Some(json!({
            "kind": "mekaneck_result", "ok": false, "evaluations": [],
            "diagnostics": [{ "severity": "error", "message": message, "line": line, "column": column }],
        })),
        residue: 1.0,
        completed: true,
        error: Some(message),
    }
}

fn run(source: &str, floors: &FloorValues, cells: &[(String, String)]) -> ActResult {
    let prog = match lang::parse(source) {
        Ok(p) => p,
        Err(e) => return refused(source, &e),
    };
    if let Err(e) = lang::typecheck(&prog, floors) {
        return refused(source, &e);
    }
    let mut sub = lang::FixedSubstrate::new();
    for (catalyst, cell) in cells {
        sub = sub.with(catalyst, cell);
    }
    let mut evaluations = Vec::new();
    for l in prog.lets() {
        match lang::eval_seek(&l.seek, &sub) {
            Ok(ev) => evaluations.push(json!({ "binding": l.name, "evaluation": ev })),
            Err(e) => return refused(source, &e),
        }
    }
    let summary: Vec<String> = evaluations
        .iter()
        .map(|e| {
            let outcome = &e["evaluation"]["outcome"];
            let what = outcome["outcome"].as_str().unwrap_or("?");
            format!("{}: {what}", e["binding"].as_str().unwrap_or("?"))
        })
        .collect();
    ActResult::done(
        json!({
            "kind": "mekaneck_result", "ok": true,
            "summary": format!("mekaneck: {}", summary.join(", ")),
            "evaluations": evaluations,
        }),
        0.0,
    )
}

/// The module (stateless).
#[derive(Debug, Default)]
pub struct Mekaneck;

impl Mekaneck {
    /// New module value.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Mekaneck {
    fn id(&self) -> &str {
        ID
    }

    fn describe(&self) -> Descriptor {
        Descriptor {
            id: ID.into(),
            description: "Mekaneck — check and run .mck inquiries: a `seek` must exclude as well as describe, rest on \
                          at least three mutually independent catalysts, and terminate by closure. A run evaluates \
                          every seek against caller-supplied cells and returns Resolved or Declined (a declination \
                          is a result) with its trace."
                .into(),
            instructions: vec![
                r#"dispatch("mekaneck", "demo")"#.into(),
                r#"dispatch("mekaneck", { kind: "check", source, floors: { Osc: 12.5 } })"#.into(),
                r#"dispatch("mekaneck", { kind: "run", source, cells: { spectral: "high", surrogate: "high", phase: "mixed" } })"#.into(),
            ],
            dsl: Some("mekaneck".into()),
            binding: BindingKind::Native,
        }
    }

    fn execute(&mut self, instruction: &Instruction, _act_budget: u32) -> ActResult {
        const EXPECTED: &str = ".mck source, \"demo\", or { kind: \"check\" | \"run\", source, floors?, cells? }";
        let cells_demo: Vec<(String, String)> = DEMO_CELLS.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
        let (kind, source) = match instruction {
            Value::String(s) if s == "demo" => return run(DEMO_SOURCE, &FloorValues::new(), &cells_demo),
            Value::String(s) => ("check", s.as_str()),
            Value::Object(_) => match (field_str(instruction, "kind").unwrap_or("check"), field_str(instruction, "source")) {
                (k @ ("check" | "run"), Some(s)) => (k, s),
                _ => return ActResult::invalid(ID, EXPECTED),
            },
            _ => return ActResult::invalid(ID, EXPECTED),
        };
        let floors = match floors_of(instruction) {
            Ok(f) => f,
            Err(m) => return ActResult::invalid(ID, &m),
        };
        if kind == "run" {
            let mut cells = Vec::new();
            if let Some(obj) = instruction.get("cells").and_then(Value::as_object) {
                for (c, v) in obj {
                    match v.as_str() {
                        Some(cell) => cells.push((c.clone(), cell.to_string())),
                        None => return ActResult::invalid(ID, "cells must be { catalyst: \"cell\" }"),
                    }
                }
            }
            return run(source, &floors, &cells);
        }
        let diags = lang::diagnose(source, &floors);
        let errors = diags.iter().filter(|d| d.severity == Severity::Error).count();
        let delta = json!({
            "kind": "mekaneck_check", "ok": errors == 0,
            "summary": format!("mekaneck: {errors} error(s), {} warning(s)", diags.len() - errors),
            "diagnostics": diagnostics_json(source, &diags),
        });
        if errors == 0 {
            ActResult::done(delta, 0.0)
        } else {
            let first = diags.iter().find(|d| d.severity == Severity::Error).map(|d| d.message.clone());
            ActResult { ok: false, output_delta: Some(delta), residue: errors as f64, completed: true, error: first }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_readme_runs_reproduce() {
        // README: all high → `regime: resolved high (record 1)`.
        let r = Mekaneck::new().execute(&json!("demo"), 1);
        assert!(r.ok, "{:?}", r.error);
        let ev = &r.output_delta.unwrap()["evaluations"][0]["evaluation"];
        assert_eq!(ev["outcome"], json!({ "outcome": "resolved", "cell": "high" }), "{ev}");
        assert_eq!(ev["record"], json!(1));
        // README: phase=mixed → `regime: declined, 2 incompatible cells` — and ok, not a failure.
        let d = Mekaneck::new().execute(
            &json!({ "kind": "run", "source": DEMO_SOURCE, "cells": { "spectral": "high", "surrogate": "high", "phase": "mixed" } }),
            1,
        );
        assert!(d.ok, "a declination is a result");
        let outcome = &d.output_delta.unwrap()["evaluations"][0]["evaluation"]["outcome"];
        assert_eq!(outcome["outcome"], json!("declined"), "{outcome}");
        let cells = &outcome["cells"];
        assert_eq!(cells.as_array().map(Vec::len), Some(2), "{cells}");
    }

    #[test]
    fn the_language_rules_are_the_front_ends() {
        assert!(validate(DEMO_SOURCE).ok);
        // A seek without `excluding` is a parse error (Thm 4.3).
        let no_excl = DEMO_SOURCE.replace("  excluding   all_other_states()\n", "");
        assert!(no_excl != DEMO_SOURCE);
        let v = validate(&no_excl);
        assert!(!v.ok);
        assert!(v.errors[0].line.is_some());
        // Fewer than three catalysts is a type error (Thm 6.2).
        let two = DEMO_SOURCE.replace("(spectral, surrogate, phase)", "(spectral, surrogate)");
        assert!(!validate(&two).ok);
        // A missing cell is a runtime refusal, not a panic.
        let r = Mekaneck::new().execute(&json!({ "kind": "run", "source": DEMO_SOURCE, "cells": { "spectral": "high" } }), 1);
        assert!(!r.ok);
    }
}
