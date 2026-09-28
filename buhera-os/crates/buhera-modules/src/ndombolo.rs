//! `ndombolo` — the deterministic Turbulance runtime (specification
//! `specs/ndombolo.md`). Wraps vendored `ndombolo-core`; reimplements nothing.
//!
//! Every act runs in a **fresh** `Session`: a dispatch is one complete script
//! or document, exactly as `ndombolo run` evaluates it, without the CLI's two
//! side effects (rewriting the `.ndo` in place, appending to `.ndo.record`).
//! The rewritten document is returned in the delta instead.

use buhera_registry::{
    field_str, instruction_kind, ActResult, BindingKind, Descriptor, DslEntry, DslError, Instruction, Module,
    Validation,
};
use ndombolo_core::doc::{format_result, Document};
use ndombolo_core::graph::GraphBuilder;
use ndombolo_core::session::{split_cells, CellResult, StoreChange};
use ndombolo_core::{parse, Session};
use serde_json::{json, Map, Value};

/// Registry id.
pub const ID: &str = "ndombolo";

/// `prototype/examples/signal.tb` — the manuscript's worked example.
pub const DEMO: &str = r#"// Worked example from the manuscript, Section 9.
// Cells are separated by a line beginning with "// ---".

item threshold = 0.7
item readings = [0.4, 0.8, 0.9]

// ---

funxn above(xs, t):
    item hits = 0
    considering all x in xs:
        given x > t:
            hits = hits + 1
    return hits

// ---

proposition Signal:
    motion Strong("enough readings clear bar")
    given above(readings, threshold) >= 2:
        support Strong with_confidence(0.9)
"#;

/// The module.
#[derive(Debug, Default)]
pub struct Ndombolo;

impl Ndombolo {
    /// New module value (stateless between acts).
    pub fn new() -> Self {
        Self
    }
}

fn cell_json(r: &CellResult) -> Value {
    let mut delta = Map::new();
    for (name, change) in &r.store_delta {
        let v = match change {
            StoreChange::Added { value, tag } => json!({ "change": "added", "value": value, "tag": tag }),
            StoreChange::Updated { from, value, tag } => {
                json!({ "change": "updated", "from": from, "value": value, "tag": tag })
            }
        };
        delta.insert(name.clone(), v);
    }
    json!({
        "index": r.index,
        "first_line": r.first_line,
        "ok": r.ok,
        "error": r.error.as_ref().map(|e| json!({ "phase": e.phase, "message": e.message, "line": e.line })),
        "output": r.output,
        "store_delta": Value::Object(delta),
        "trace": r.trace,
    })
}

fn propositions_json(s: &Session) -> Value {
    Value::Array(
        s.compiler
            .propositions
            .iter()
            .map(|(name, p)| {
                json!({
                    "name": name,
                    "motions": p.motions.iter().map(|m| json!({
                        "name": m.name, "text": m.text, "score": p.score(&m.name)
                    })).collect::<Vec<_>>(),
                    "verdicts": p.verdicts.iter().map(|v| json!({
                        "motion": v.motion, "stance": v.stance, "confidence": v.confidence, "line": v.line
                    })).collect::<Vec<_>>(),
                })
            })
            .collect(),
    )
}

fn store_json(s: &Session) -> Value {
    let mut m = Map::new();
    for (k, v) in s.store() {
        m.insert(k, v);
    }
    Value::Object(m)
}

/// Static contact graph over cells that parse (needs no execution).
fn graph_json(cells: &[(usize, String)]) -> (Value, Vec<usize>) {
    let mut b = GraphBuilder::new();
    let mut unparsed = Vec::new();
    for (stage, (first_line, text)) in cells.iter().enumerate() {
        match parse(text, *first_line) {
            Ok(body) => b.add_cell(&body, stage),
            Err(_) => unparsed.push(stage),
        }
    }
    (b.graph.to_json(), unparsed)
}

/// Run cells in order in a fresh session, stopping at the first failure.
fn run_cells(cells: &[(usize, String)]) -> (Session, Vec<CellResult>, Option<usize>) {
    let mut s = Session::new();
    let mut out = Vec::new();
    let mut stopped = None;
    for (i, (first_line, text)) in cells.iter().enumerate() {
        let r = s.run_cell(i, *first_line, text);
        let ok = r.ok;
        out.push(r);
        if !ok {
            stopped = Some(i);
            break;
        }
    }
    (s, out, stopped)
}

fn result_delta(
    cells: &[(usize, String)],
    session: &Session,
    results: &[CellResult],
    stopped: Option<usize>,
    document: Option<String>,
) -> (Value, usize) {
    let events: usize = results.iter().map(|r| r.trace.len()).sum();
    let (graph, unparsed) = graph_json(cells);
    let summary = match stopped {
        None => format!("ndombolo: ran {} cell(s), {events} event(s)", results.len()),
        Some(i) => format!(
            "ndombolo: stopped at cell {i} of {} — {}",
            cells.len(),
            results[i].error.as_ref().map(|e| e.message.as_str()).unwrap_or("error")
        ),
    };
    let mut delta = json!({
        "kind": "ndombolo_result",
        "summary": summary,
        "cells": results.iter().map(cell_json).collect::<Vec<_>>(),
        "stopped_at": stopped,
        "store": store_json(session),
        "propositions": propositions_json(session),
        "graph": graph,
        "cells_that_did_not_parse": unparsed,
        "events": events,
    });
    if let Some(doc) = document {
        delta["document"] = Value::String(doc);
    }
    (delta, events)
}

/// Cells of a `.ndo` document as (first_line, text), with their block indices.
fn doc_cells(doc: &Document) -> Vec<(usize, usize, String)> {
    doc.cell_indices()
        .into_iter()
        .filter_map(|b| doc.cell_text(b).map(|t| (b, doc.line_of(b) + 1, t)))
        .collect()
}

impl Module for Ndombolo {
    fn id(&self) -> &str {
        ID
    }

    fn describe(&self) -> Descriptor {
        Descriptor {
            id: ID.into(),
            description: "ndombolo — the deterministic Turbulance runtime from kwasa-kwasa: run cell-separated \
                          Turbulance source or a .ndo notebook, returning per-cell store deltas, the rule trace, \
                          proposition scores and the static contact graph. No clock, no randomness, no model."
                .into(),
            instructions: vec![
                r#"dispatch("ndombolo", "demo")"#.into(),
                r#"dispatch("ndombolo", "item x = 2\n// ---\nprint(x * 3)")"#.into(),
                r#"dispatch("ndombolo", { kind: "ndo", document: "<.ndo markdown>" })"#.into(),
                r#"dispatch("ndombolo", { kind: "graph", source })"#.into(),
            ],
            dsl: Some("turbulance".into()),
            binding: BindingKind::Native,
        }
    }

    fn execute(&mut self, instruction: &Instruction, _act_budget: u32) -> ActResult {
        const EXPECTED: &str = "Turbulance source, \"demo\", or { kind: run|ndo|graph|validate, … }";
        let kind = instruction_kind(instruction);
        let is_obj = instruction.is_object();
        let source: Option<String> = match (kind, is_obj) {
            (None, false) if instruction.is_null() => Some(DEMO.to_string()),
            (Some("demo") | Some(""), false) => Some(DEMO.to_string()),
            (Some(s), false) => Some(s.to_string()),
            (Some("run" | "graph" | "validate"), true) => field_str(instruction, "source").map(str::to_string),
            _ => None,
        };

        match (kind, is_obj) {
            (Some("ndo"), true) => {
                let Some(text) = field_str(instruction, "document") else {
                    return ActResult::invalid(ID, "{ kind: \"ndo\", document: string }");
                };
                let mut doc = Document::parse(text);
                let cells = doc_cells(&doc);
                let flat: Vec<(usize, String)> = cells.iter().map(|(_, l, t)| (*l, t.clone())).collect();
                let (session, results, stopped) = run_cells(&flat);
                for (r, (block, _, _)) in results.iter().zip(cells.iter()) {
                    doc.splice(*block, &format_result(&r.output, &r.store_delta, r.error.as_ref()));
                }
                let (delta, events) = result_delta(&flat, &session, &results, stopped, Some(doc.render()));
                return ActResult::done(delta, events as f64);
            }
            (Some("graph"), true) => {
                let Some(src) = source else { return ActResult::invalid(ID, "{ kind: \"graph\", source: string }") };
                let (graph, unparsed) = graph_json(&split_cells(&src));
                let n = graph["item_count"].as_u64().unwrap_or(0) + graph["contact_count"].as_u64().unwrap_or(0);
                return ActResult::done(
                    json!({ "kind": "ndombolo_graph", "summary": format!("ndombolo graph: {n} items+contacts"),
                            "graph": graph, "cells_that_did_not_parse": unparsed }),
                    n as f64,
                );
            }
            (Some("validate"), true) => {
                let Some(src) = source else { return ActResult::invalid(ID, "{ kind: \"validate\", source: string }") };
                let v = validate(&src);
                return ActResult::done(json!({ "kind": "dsl_validation", "dsl": "turbulance", "ok": v.ok, "errors": v.errors }), v.errors.len() as f64);
            }
            _ => {}
        }

        let Some(src) = source else { return ActResult::invalid(ID, EXPECTED) };
        let cells = split_cells(&src);
        let (session, results, stopped) = run_cells(&cells);
        let (delta, events) = result_delta(&cells, &session, &results, stopped, None);
        // ok = the act ran the script. A stop at a failing cell is a normal,
        // reported outcome (tutorials stop by design) and is visible in
        // `stopped_at`; residue counts the trace events deposited (spec §Residue).
        ActResult::done(delta, events as f64)
    }
}

/// The Turbulance front end: lex + parse every cell (never evaluates).
pub fn validate(source: &str) -> Validation {
    let errors: Vec<DslError> = split_cells(source)
        .iter()
        .filter_map(|(first_line, text)| parse(text, *first_line).err())
        .map(|e| DslError::at(e.message.clone(), e.line as u32, None))
        .collect();
    if errors.is_empty() {
        Validation::valid()
    } else {
        Validation::invalid(errors)
    }
}

/// DSL registry entry.
pub fn dsl() -> DslEntry {
    DslEntry {
        id: "turbulance",
        label: "Turbulance (ndombolo)",
        extension: ".tb",
        module_id: ID,
        pack_id: "turbulance",
        validate,
    }
}
