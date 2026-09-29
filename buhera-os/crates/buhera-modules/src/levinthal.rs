//! `levinthal` — protein folding as backward trajectory completion
//! (specification `specs/levinthal.md`). Wraps vendored `levinthal-core`:
//! partition states (n, l, m, s) under the selection rules, the trajectory
//! derived backward from a goal state to the origin, and per-residue
//! S-entropy coordinates of a sequence.
//!
//! Deliberately not exposed: the ternary-string projections (upstream's
//! `to_sentropy` returns a constant placeholder and `to_cell_bounds` ignores
//! the trit values, U-lev-4), `coherence_proxy` presented as a physical
//! quantity, and molecular weight (upstream sums free amino-acid average
//! masses, which is wrong for a peptide; U-lev-2).

use buhera_registry::{field_str, instruction_kind, ActResult, BindingKind, Descriptor, Instruction, Module};
use levinthal_core::amino_acid::parse_sequence;
use levinthal_core::{PartitionState, SEntropyCoord, Trajectory};
use serde_json::{json, Value};

/// Registry id.
pub const ID: &str = "levinthal";

/// The module (stateless).
#[derive(Debug, Default)]
pub struct Levinthal;

impl Levinthal {
    /// New module value.
    pub fn new() -> Self {
        Self
    }
}

fn state_json(s: &PartitionState) -> Value {
    json!({ "n": s.n(), "l": s.l(), "m": s.m(), "s": s.s().value() })
}

fn coord_json(c: &SEntropyCoord) -> Value {
    json!({ "sk": c.sk(), "st": c.st(), "se": c.se() })
}

fn complete(goal: PartitionState) -> ActResult {
    let t = Trajectory::complete(goal);
    ActResult::done(
        json!({
            "kind": "levinthal_trajectory",
            "summary": format!("levinthal: {} state(s) from the origin to ({}, {}, {}, {:+})",
                               t.len(), goal.n(), goal.l(), goal.m(), goal.s().value()),
            "goal": state_json(&goal),
            "states": t.states().iter().map(state_json).collect::<Vec<_>>(),
            "continuous": t.is_continuous(),
            "coherence_profile": t.coherence_profile(),
            "max_depth": t.max_depth(),
            "max_complexity": t.max_complexity(),
        }),
        0.0,
    )
}

fn analyze(sequence: &str) -> ActResult {
    let residues = match parse_sequence(sequence) {
        Ok(r) if !r.is_empty() => r,
        Ok(_) => return ActResult::invalid(ID, "{ kind: \"analyze\", sequence: <non-empty one-letter sequence> }"),
        Err(e) => return ActResult::fail(&[&format!("levinthal: {e}")], e.to_string()),
    };
    let n = residues.len() as f64;
    let coords: Vec<SEntropyCoord> = residues.iter().map(|a| a.sentropy()).collect();
    let mean = |f: fn(&SEntropyCoord) -> f64| coords.iter().map(f).sum::<f64>() / n;
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for a in &residues {
        *counts.entry(format!("{:?}", a.category())).or_default() += 1;
    }
    ActResult::done(
        json!({
            "kind": "levinthal_analysis",
            "summary": format!("levinthal: {} residue(s)", residues.len()),
            "length": residues.len(),
            "net_charge": residues.iter().map(|a| a.charge()).sum::<f64>(),
            "category_counts": counts,
            "mean_sentropy": { "sk": mean(SEntropyCoord::sk), "st": mean(SEntropyCoord::st), "se": mean(SEntropyCoord::se) },
            "residues": residues.iter().zip(&coords).map(|(a, c)| json!({
                "code": a.code().to_string(), "code3": a.code3(), "category": format!("{:?}", a.category()),
                "hydrophobicity": a.hydrophobicity(), "charge": a.charge(), "sentropy": coord_json(c),
            })).collect::<Vec<_>>(),
        }),
        0.0,
    )
}

impl Module for Levinthal {
    fn id(&self) -> &str {
        ID
    }

    fn describe(&self) -> Descriptor {
        Descriptor {
            id: ID.into(),
            description: "levinthal — folding as backward completion: derive the trajectory from a goal partition \
                          state (n, l, m, s) back to the origin under the selection rules, with its coherence \
                          profile; count state capacities; map a protein sequence to per-residue S-entropy \
                          coordinates."
                .into(),
            instructions: vec![
                r#"dispatch("levinthal", "demo")"#.into(),
                r#"dispatch("levinthal", { kind: "complete", goal: { n: 3, l: 2, m: 1, s: 0.5 } })"#.into(),
                r#"dispatch("levinthal", { kind: "analyze", sequence: "MKTAYIAKQRQISFVKSHFSRQ" })"#.into(),
                r#"dispatch("levinthal", { kind: "capacity", depth: 4 })"#.into(),
            ],
            dsl: None,
            binding: BindingKind::Native,
        }
    }

    fn execute(&mut self, instruction: &Instruction, _act_budget: u32) -> ActResult {
        const EXPECTED: &str = "\"demo\" or { kind: complete|analyze|capacity, … }";
        match (instruction_kind(instruction), instruction.is_object()) {
            (Some("demo"), false) => match PartitionState::new(3, 2, 1, 0.5) {
                Ok(g) => complete(g),
                Err(e) => ActResult::fail(&[&e.to_string()], e.to_string()),
            },
            (Some("complete"), true) => {
                let g = instruction.get("goal").unwrap_or(&Value::Null);
                let int = |k: &str| g.get(k).and_then(Value::as_i64);
                let (Some(n), Some(l), Some(m), Some(s)) = (int("n"), int("l"), int("m"), g.get("s").and_then(Value::as_f64)) else {
                    return ActResult::invalid(ID, "{ kind: \"complete\", goal: { n ≥ 1, 0 ≤ l < n, |m| ≤ l, s: ±0.5 } }");
                };
                if n < 1 || l < 0 || n > u32::MAX as i64 || l > u32::MAX as i64 || m.abs() > i32::MAX as i64 {
                    return ActResult::fail(&["levinthal: n ≥ 1 and l ≥ 0 are required"], "invalid partition coordinates");
                }
                match PartitionState::new(n as u32, l as u32, m as i32, s) {
                    Ok(goal) => complete(goal),
                    Err(e) => ActResult::fail(&[&format!("levinthal: {e}")], e.to_string()),
                }
            }
            (Some("analyze"), true) => match field_str(instruction, "sequence") {
                Some(seq) => analyze(seq),
                None => ActResult::invalid(ID, "{ kind: \"analyze\", sequence }"),
            },
            (Some("capacity"), true) => match instruction.get("depth").and_then(Value::as_u64) {
                Some(d) if (1..=64).contains(&d) => ActResult::done(
                    json!({ "kind": "levinthal_capacity", "depth": d,
                            "capacity": PartitionState::capacity(d as u32),
                            "cumulative": PartitionState::cumulative_capacity(d as u32) }),
                    0.0,
                ),
                _ => ActResult::invalid(ID, "{ kind: \"capacity\", depth: 1..=64 }"),
            },
            _ => ActResult::invalid(ID, EXPECTED),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_demo_reproduces_the_cli_trajectory() {
        // levinthal CLI: complete -n 3 -l 2 -m 1 →
        // (1,0,0,+½)→(2,0,0,+½)→(3,0,0,+½)→(3,1,1,+½)→(3,2,1,+½)
        let r = Levinthal::new().execute(&json!("demo"), 1);
        assert!(r.ok);
        let d = r.output_delta.unwrap();
        let got: Vec<(u64, u64, i64)> = d["states"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| (s["n"].as_u64().unwrap(), s["l"].as_u64().unwrap(), s["m"].as_i64().unwrap()))
            .collect();
        assert_eq!(got, vec![(1, 0, 0), (2, 0, 0), (3, 0, 0), (3, 1, 1), (3, 2, 1)]);
        assert_eq!(d["continuous"], json!(true));
    }

    #[test]
    fn selection_rules_and_sequences_are_the_engines() {
        let bad = Levinthal::new().execute(&json!({ "kind": "complete", "goal": { "n": 2, "l": 2, "m": 0, "s": 0.5 } }), 1);
        assert!(!bad.ok, "l < n is required");
        let x = Levinthal::new().execute(&json!({ "kind": "analyze", "sequence": "XYZ" }), 1);
        assert!(!x.ok, "X is not an amino acid");
        let ok = Levinthal::new().execute(&json!({ "kind": "analyze", "sequence": "PEPTIDE" }), 1);
        assert_eq!(ok.output_delta.unwrap()["length"], json!(7));
        let cap = Levinthal::new().execute(&json!({ "kind": "capacity", "depth": 3 }), 1);
        assert_eq!(cap.output_delta.unwrap()["capacity"], json!(18)); // C(n) = 2n²
    }
}
