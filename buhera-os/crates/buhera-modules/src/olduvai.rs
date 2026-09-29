//! `olduvai` — intrinsic addressing from the Olduvai exchange (specification
//! `specs/olduvai.md`). Wraps vendored `olduvai-core`: S-coordinates in
//! [0, 1]³ become ternary addresses (one trit per axis-refinement, depth ≤
//! 12), a prefix trie answers nearest-neighbour queries with an honest
//! fallback, and attention is water-filled across scenes.
//!
//! The module keeps one trie as state (R6): `insert` adds labelled points,
//! `nearest` and `ranked` query them. A fallback that had to back off is
//! reported, never hidden: the act residue of `nearest` is the number of
//! trits of resolution it gave up.

use buhera_registry::{field_str, instruction_kind, ActResult, BindingKind, Descriptor, Instruction, Module};
use olduvai_core::agent::{self, Scene};
use olduvai_core::foreman::{self, Cycle};
use olduvai_core::{Address, Coordinates, Trie, FULL_DEPTH};
use serde_json::{json, Value};

/// Registry id.
pub const ID: &str = "olduvai";

/// The module: one trie of labelled addresses.
#[derive(Debug, Default)]
pub struct Olduvai {
    trie: Trie<String>,
}

impl Olduvai {
    /// New module value with an empty trie.
    pub fn new() -> Self {
        Self { trie: Trie::new() }
    }
}

fn coords_of(v: &Value) -> Result<Coordinates, String> {
    let get = |k: &str| v.get(k).and_then(Value::as_f64).ok_or_else(|| format!("missing number `{k}`"));
    Coordinates::new(get("s_k")?, get("s_t")?, get("s_e")?).map_err(|e| e.to_string())
}

fn depth_of(instruction: &Instruction) -> Result<usize, String> {
    match instruction.get("depth") {
        None | Some(Value::Null) => Ok(FULL_DEPTH),
        Some(v) => match v.as_u64() {
            Some(d) if (d as usize) <= FULL_DEPTH => Ok(d as usize),
            _ => Err(format!("depth must be an integer in 0..={FULL_DEPTH}")),
        },
    }
}

/// An address from `address: "…"` or from `coords: {s_k, s_t, s_e}` at `depth`.
fn address_of(instruction: &Instruction) -> Result<Address, String> {
    if let Some(s) = field_str(instruction, "address") {
        return s.parse::<Address>().map_err(|e| e.to_string());
    }
    let c = instruction.get("coords").ok_or("give `address` or `coords: { s_k, s_t, s_e }`")?;
    Ok(Address::encode(coords_of(c)?, depth_of(instruction)?))
}

fn refuse(message: String) -> ActResult {
    ActResult::fail(&[&format!("olduvai: {message}")], message)
}

impl Olduvai {
    fn insert(&mut self, entries: &[Value]) -> ActResult {
        let mut added = Vec::new();
        for e in entries {
            let Some(label) = e.get("label").and_then(Value::as_str) else {
                return refuse("each entry needs a `label`".into());
            };
            let addr = match address_of(e) {
                Ok(a) => a,
                Err(m) => return refuse(format!("entry {label:?}: {m}")),
            };
            self.trie.insert(&addr, label.to_string());
            added.push(json!({ "label": label, "address": addr.to_string() }));
        }
        ActResult::done(
            json!({ "kind": "olduvai_trie", "summary": format!("olduvai: {} inserted, {} in trie", added.len(), self.trie.len()),
                    "inserted": added, "size": self.trie.len() }),
            0.0,
        )
    }

    fn nearest(&self, addr: &Address) -> ActResult {
        let f = self.trie.nearest(addr);
        let lost = f.resolution_lost();
        ActResult::done(
            json!({
                "kind": "olduvai_nearest",
                "summary": if f.is_exact() { format!("olduvai: exact at depth {}", f.requested_depth) }
                           else { format!("olduvai: backed off {lost} trit(s) to depth {}", f.matched_depth) },
                "query": addr.to_string(), "prefix": f.address.to_string(),
                "matched_depth": f.matched_depth, "requested_depth": f.requested_depth,
                "exact": f.is_exact(), "resolution_lost": lost, "values": f.values,
            }),
            // Residue: the trits of resolution this answer gave up.
            lost as f64,
        )
    }
}

impl Module for Olduvai {
    fn id(&self) -> &str {
        ID
    }

    fn describe(&self) -> Descriptor {
        Descriptor {
            id: ID.into(),
            description: "Olduvai — intrinsic addressing: encode S-coordinates (s_k, s_t, s_e) ∈ [0,1]³ to ternary \
                          addresses and back, measure similarity by shared prefix, keep a trie of labelled points and \
                          answer nearest/ranked queries with an explicit fallback; water-fill attention across scenes; \
                          check the coherence of a closed foreman cycle."
                .into(),
            instructions: vec![
                r#"dispatch("olduvai", "demo")"#.into(),
                r#"dispatch("olduvai", { kind: "encode", coords: { s_k: 0.2, s_t: 0.7, s_e: 0.5 }, depth: 8 })"#.into(),
                r#"dispatch("olduvai", { kind: "insert", entries: [{ label: "maize", coords: { s_k: 0.3, s_t: 0.6, s_e: 0.2 } }] })"#.into(),
                r#"dispatch("olduvai", { kind: "nearest", coords: { s_k: 0.31, s_t: 0.6, s_e: 0.2 } })"#.into(),
                r#"dispatch("olduvai", { kind: "water_fill", scenes: [{ name: "a", gain_k: 2 }], budget: 1 })"#.into(),
            ],
            dsl: None,
            binding: BindingKind::Native,
        }
    }

    fn execute(&mut self, instruction: &Instruction, _act_budget: u32) -> ActResult {
        const EXPECTED: &str = "\"demo\" or { kind: encode|decode|compare|insert|nearest|ranked|water_fill|check_cycle|reset, … }";
        match (instruction_kind(instruction), instruction.is_object()) {
            (Some("demo"), false) => {
                self.trie = Trie::new();
                let pts = [("maize", 0.30, 0.60, 0.20), ("sorghum", 0.32, 0.61, 0.22), ("cassava", 0.80, 0.10, 0.70)];
                let entries: Vec<Value> = pts
                    .iter()
                    .map(|(l, k, t, e)| json!({ "label": l, "coords": { "s_k": k, "s_t": t, "s_e": e } }))
                    .collect();
                let inserted = self.insert(&entries);
                if !inserted.ok {
                    return inserted;
                }
                match address_of(&json!({ "coords": { "s_k": 0.31, "s_t": 0.60, "s_e": 0.21 } })) {
                    Ok(q) => self.nearest(&q),
                    Err(m) => refuse(m),
                }
            }
            (Some("reset"), false) => {
                self.trie = Trie::new();
                ActResult::done(json!({ "kind": "olduvai_trie", "summary": "olduvai: trie cleared", "size": 0 }), 0.0)
            }
            (Some("encode"), true) => match address_of(instruction) {
                Ok(a) => ActResult::done(json!({ "kind": "olduvai_address", "address": a.to_string(), "depth": a.depth() }), 0.0),
                Err(m) => refuse(m),
            },
            (Some("decode"), true) => match field_str(instruction, "address").map(str::parse::<Address>) {
                Some(Ok(a)) => {
                    let [s_k, s_t, s_e] = a.decode().as_array();
                    ActResult::done(json!({ "kind": "olduvai_coords", "address": a.to_string(), "depth": a.depth(),
                                            "centre": { "s_k": s_k, "s_t": s_t, "s_e": s_e } }), 0.0)
                }
                Some(Err(e)) => refuse(e.to_string()),
                None => ActResult::invalid(ID, "{ kind: \"decode\", address }"),
            },
            (Some("compare"), true) => {
                match (field_str(instruction, "a").map(str::parse::<Address>), field_str(instruction, "b").map(str::parse::<Address>)) {
                    (Some(Ok(a)), Some(Ok(b))) => ActResult::done(
                        json!({ "kind": "olduvai_similarity", "a": a.to_string(), "b": b.to_string(), "shared_prefix": a.common_prefix_len(&b) }),
                        0.0,
                    ),
                    (Some(Err(e)), _) | (_, Some(Err(e))) => refuse(e.to_string()),
                    _ => ActResult::invalid(ID, "{ kind: \"compare\", a, b }"),
                }
            }
            (Some("insert"), true) => match instruction.get("entries").and_then(Value::as_array) {
                Some(entries) => self.insert(entries),
                None => ActResult::invalid(ID, "{ kind: \"insert\", entries: [{ label, address | coords }] }"),
            },
            (Some("nearest"), true) => match address_of(instruction) {
                Ok(a) => self.nearest(&a),
                Err(m) => refuse(m),
            },
            (Some("ranked"), true) => match address_of(instruction) {
                Ok(a) => {
                    let ranked: Vec<Value> = self
                        .trie
                        .ranked(&a)
                        .into_iter()
                        .map(|r| json!({ "label": r.value, "address": r.address.to_string(), "shared_prefix": r.shared_prefix }))
                        .collect();
                    ActResult::done(json!({ "kind": "olduvai_ranked", "query": a.to_string(), "ranked": ranked }), 0.0)
                }
                Err(m) => refuse(m),
            },
            (Some("water_fill"), true) => {
                let scenes: Result<Vec<Scene>, _> =
                    serde_json::from_value(instruction.get("scenes").cloned().unwrap_or(Value::Null));
                let budget = instruction.get("budget").and_then(Value::as_f64);
                match (scenes, budget) {
                    (Ok(s), Some(b)) if b.is_finite() && b >= 0.0 => {
                        let wf = agent::water_fill(&s, b);
                        let mut v = serde_json::to_value(&wf).unwrap_or(Value::Null);
                        v["kind"] = json!("olduvai_water_fill");
                        ActResult::done(v, 0.0)
                    }
                    _ => ActResult::invalid(ID, "{ kind: \"water_fill\", scenes: [{ name, gain_k > 0 }], budget ≥ 0 }"),
                }
            }
            (Some("check_cycle"), true) => {
                match serde_json::from_value::<Cycle>(instruction.get("cycle").cloned().unwrap_or(Value::Null)) {
                    Ok(c) => {
                        let mut v = serde_json::to_value(foreman::check_cycle(&c)).unwrap_or(Value::Null);
                        let v = if v.is_object() {
                            v["kind"] = json!("olduvai_cycle");
                            v
                        } else {
                            json!({ "kind": "olduvai_cycle", "result": v })
                        };
                        ActResult::done(v, 0.0)
                    }
                    Err(e) => refuse(format!("cycle: {e}")),
                }
            }
            _ => ActResult::invalid(ID, EXPECTED),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trips_to_the_cell_centre() {
        let mut m = Olduvai::new();
        let r = m.execute(&json!({ "kind": "encode", "coords": { "s_k": 0.2, "s_t": 0.7, "s_e": 0.5 }, "depth": 6 }), 1);
        let addr = r.output_delta.unwrap()["address"].as_str().unwrap().to_string();
        let d = m.execute(&json!({ "kind": "decode", "address": addr }), 1).output_delta.unwrap();
        for (k, want) in [("s_k", 0.2), ("s_t", 0.7), ("s_e", 0.5)] {
            let got = d["centre"][k].as_f64().unwrap();
            // Depth 6 interleaves the three axes: two trits each, a cell 1/9 wide.
            assert!((got - want).abs() <= 0.5 / 9.0 + 1e-12, "{k}: {got} vs {want}");
        }
    }

    #[test]
    fn nearest_reports_the_resolution_it_gave_up() {
        let mut m = Olduvai::new();
        let r = m.execute(&json!("demo"), 1);
        assert!(r.ok, "{:?}", r.error);
        let d = r.output_delta.unwrap();
        assert_eq!(d["resolution_lost"].as_u64().unwrap() as f64, r.residue);
        let vals: Vec<&str> = d["values"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
        assert!(!vals.contains(&"cassava"), "a distant point is never the nearest: {vals:?}");
        // An exact hit loses nothing.
        let exact = m.execute(&json!({ "kind": "nearest", "coords": { "s_k": 0.30, "s_t": 0.60, "s_e": 0.20 } }), 1);
        assert_eq!(exact.residue, 0.0);
    }

    #[test]
    fn out_of_range_coordinates_are_refused() {
        let mut m = Olduvai::new();
        let r = m.execute(&json!({ "kind": "encode", "coords": { "s_k": 1.5, "s_t": 0.0, "s_e": 0.0 } }), 1);
        assert!(!r.ok);
    }
}
