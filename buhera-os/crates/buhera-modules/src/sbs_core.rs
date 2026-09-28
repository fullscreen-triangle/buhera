//! `sbs-core` — the Systems Biology Shaders observation calculus, Rust port
//! (specification `specs/sbs-core.md`). Wraps the vendored `hegel/sbs` crate.
//!
//! This is **not** the `sbs` module: upstream's Rust crate has no `.sbs`
//! language, and it differs numerically from the JS runtime in the
//! documented ways (chemical-potential convention, Spearman tie handling,
//! visibility edge cases). It takes an explicit circuit, an SBML document, or
//! the canonical glycolysis demo, plus explicit `(edge, factor)` perturbations.
//!
//! The engine panics on an out-of-range edge endpoint and on NaN; this
//! adapter validates both before calling it (contract I2).

use buhera_registry::{field_str, instruction_kind, ActResult, BindingKind, Descriptor, Instruction, Module};
use sbs::circuit::{Circuit, Edge, Node, Perturbation};
use sbs::metrics::find_optimal_perturbation;
use sbs::sbml::parse_sbml;
use sbs::Solver;
use serde_json::{json, Value};

/// Registry id.
pub const ID: &str = "sbs-core";

/// The module (stateless).
#[derive(Debug, Default)]
pub struct SbsCore;

impl SbsCore {
    /// New module value.
    pub fn new() -> Self {
        Self
    }
}

fn finite(v: &Value, what: &str) -> Result<f64, String> {
    match v.as_f64() {
        Some(x) if x.is_finite() => Ok(x),
        _ => Err(format!("{what} must be a finite number")),
    }
}

/// Build a circuit from `{ nodes: [{name, mu, concentration, compartment?}],
/// edges: [{src, dst, conductance?, rate?}] }` (indices into `nodes`).
fn circuit_from_json(v: &Value) -> Result<Circuit, String> {
    let nodes = v.get("nodes").and_then(Value::as_array).ok_or("circuit.nodes must be an array")?;
    let edges = v.get("edges").and_then(Value::as_array).ok_or("circuit.edges must be an array")?;
    let mut c = Circuit::new();
    for (i, n) in nodes.iter().enumerate() {
        let name = n.get("name").and_then(Value::as_str).ok_or(format!("nodes[{i}].name must be a string"))?;
        let mu = finite(n.get("mu").unwrap_or(&json!(0.0)), &format!("nodes[{i}].mu"))?;
        let conc = finite(n.get("concentration").unwrap_or(&json!(1.0)), &format!("nodes[{i}].concentration"))?;
        let mut node = Node::new(name, mu, conc);
        if let Some(comp) = n.get("compartment").and_then(Value::as_str) {
            node = node.with_compartment(comp);
        }
        c.add_node(node);
    }
    for (i, e) in edges.iter().enumerate() {
        let idx = |k: &str| -> Result<usize, String> {
            let x = e.get(k).and_then(Value::as_u64).ok_or(format!("edges[{i}].{k} must be a node index"))? as usize;
            if x >= nodes.len() {
                return Err(format!("edges[{i}].{k} = {x} is out of range ({} nodes)", nodes.len()));
            }
            Ok(x)
        };
        let (src, dst) = (idx("src")?, idx("dst")?);
        let conductance = finite(e.get("conductance").unwrap_or(&json!(1.0)), &format!("edges[{i}].conductance"))?;
        let mut edge = Edge::new(src, dst, conductance);
        if let Some(r) = e.get("rate") {
            edge = edge.with_rate(finite(r, &format!("edges[{i}].rate"))?, c.nodes[src].concentration);
        }
        c.add_edge(edge);
    }
    Ok(c)
}

fn perturbations_from_json(v: Option<&Value>, n_edges: usize) -> Result<Vec<Perturbation>, String> {
    let Some(arr) = v else { return Ok(Vec::new()) };
    let arr = arr.as_array().ok_or("perturbations must be an array of { edge, factor }")?;
    arr.iter()
        .enumerate()
        .map(|(i, p)| {
            let edge = p.get("edge").and_then(Value::as_u64).ok_or(format!("perturbations[{i}].edge must be an edge index"))?
                as usize;
            if edge >= n_edges {
                return Err(format!("perturbations[{i}].edge = {edge} is out of range ({n_edges} edges)"));
            }
            let factor = finite(p.get("factor").unwrap_or(&json!(0.1)), &format!("perturbations[{i}].factor"))?;
            Ok(Perturbation::new(edge, factor))
        })
        .collect()
}

fn circuit_of(instruction: &Instruction) -> Result<Circuit, String> {
    if let Some(xml) = field_str(instruction, "sbml") {
        return parse_sbml(xml).map_err(|e| format!("sbml: {e}"));
    }
    match instruction.get("circuit") {
        Some(c) => circuit_from_json(c),
        None => Ok(Circuit::demo_glycolysis()),
    }
}

fn observe(circuit: Circuit, perturbations: Vec<Perturbation>, restore: Option<usize>) -> ActResult {
    if circuit.nodes.iter().any(|n| !n.mu.is_finite()) {
        return ActResult::fail(&["sbs-core: a node potential is not finite"], "invalid circuit");
    }
    let n_nodes = circuit.num_nodes();
    let n_edges = circuit.num_edges();
    let solver = Solver::new(circuit.clone()).with_perturbations(perturbations.clone());
    let r = solver.solve();
    let v = r.visibility();
    let therapy = restore.map(|max| {
        find_optimal_perturbation(&circuit, &perturbations, max)
            .iter()
            .map(|p| json!({ "edge": p.edge_idx, "factor": p.factor }))
            .collect::<Vec<_>>()
    });
    let mut delta = json!({
        "kind": "sbs_core_result",
        "summary": format!("sbs-core: {n_nodes} nodes, {n_edges} edges  R={:.4}  V={:.4}", r.coherence(), v),
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "coherence": r.coherence(),
        "visibility": v,
        "metrics": r.metrics,
        "perturbations": perturbations.iter().map(|p| json!({ "edge": p.edge_idx, "factor": p.factor })).collect::<Vec<_>>(),
        "backend": r.backend,
        "compute_time_us": r.compute_time_us,
    });
    if let Some(t) = therapy {
        delta["therapy"] = Value::Array(t);
    }
    // Residue = 1 − V: how far the observed flux pattern is from the healthy
    // baseline (0 when unperturbed). R is not a distance and is not used.
    ActResult::done(delta, (1.0 - v).max(0.0))
}

impl Module for SbsCore {
    fn id(&self) -> &str {
        ID
    }

    fn describe(&self) -> Descriptor {
        Descriptor {
            id: ID.into(),
            description: "sbs-core — the SBS observation calculus (Rust port, no DSL): S-entropy triples, triple \
                          coherence R and flux visibility V for an explicit circuit, an SBML model, or the \
                          glycolysis demo, under explicit edge perturbations; optional l1 restoration."
                .into(),
            instructions: vec![
                r#"dispatch("sbs-core", "demo")"#.into(),
                r#"dispatch("sbs-core", { kind: "observe", perturbations: [{ edge: 0, factor: 0.1 }] })"#.into(),
                r#"dispatch("sbs-core", { kind: "observe", sbml: "<sbml>…</sbml>" })"#.into(),
                r#"dispatch("sbs-core", { kind: "restore", perturbations: [{ edge: 0, factor: 0.1 }], max_edges: 3 })"#
                    .into(),
            ],
            dsl: None,
            binding: BindingKind::Native,
        }
    }

    fn execute(&mut self, instruction: &Instruction, _act_budget: u32) -> ActResult {
        let (kind, obj) = (instruction_kind(instruction), instruction.is_object());
        let restore = match (kind, obj) {
            (Some("demo") | Some(""), false) => {
                return observe(Circuit::demo_glycolysis(), vec![Perturbation::new(0, 0.1)], None);
            }
            (_, false) if instruction.is_null() => {
                return observe(Circuit::demo_glycolysis(), vec![Perturbation::new(0, 0.1)], None);
            }
            (Some("observe"), true) => None,
            (Some("restore"), true) => Some(instruction.get("max_edges").and_then(Value::as_u64).unwrap_or(3) as usize),
            _ => return ActResult::invalid(ID, "\"demo\" or { kind: observe|restore, circuit?|sbml?, perturbations? }"),
        };
        let circuit = match circuit_of(instruction) {
            Ok(c) => c,
            Err(e) => return ActResult::fail(&[&format!("sbs-core: {e}")], "invalid circuit"),
        };
        let perturbations = match perturbations_from_json(instruction.get("perturbations"), circuit.num_edges()) {
            Ok(p) => p,
            Err(e) => return ActResult::fail(&[&format!("sbs-core: {e}")], "invalid perturbation"),
        };
        observe(circuit, perturbations, restore)
    }
}
