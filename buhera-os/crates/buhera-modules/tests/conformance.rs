//! Catalogue conformance (specification 05) and per-module behaviour
//! (specifications/specs/<module>.md, "Conformance" sections).
//!
//! Run the full check with `cargo test -p buhera-modules --features full`.
//! With a feature subset, C1 "not registered" findings for disabled modules
//! are expected and filtered; every other rule still applies.

use buhera_modules::{federation, Options};
use buhera_registry::{conformance, Catalogue, Host};
use serde_json::{json, Value};

fn catalogue() -> Catalogue {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../specifications/registry/catalogue.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    Catalogue::from_json(&text).expect("catalogue parses")
}

fn enabled(id: &str) -> bool {
    match id {
        "vahera" => cfg!(feature = "vahera"),
        "ndombolo" => cfg!(feature = "ndombolo"),
        "windtunnel" => cfg!(feature = "windtunnel"),
        "tracker" => cfg!(feature = "tracker"),
        "sbs-core" => cfg!(feature = "sbs-core"),
        "zangalewa-dsl" => cfg!(feature = "zangalewa"),
        _ => false,
    }
}

#[test]
fn rust_federation_conforms_to_the_catalogue() {
    let cat = catalogue();
    let (modules, dsls) = federation(Options::default());
    let violations: Vec<String> = conformance(&cat, Host::Rust, &modules, &dsls)
        .into_iter()
        .filter(|v| {
            // Disabled-feature modules and their languages are legitimately absent.
            let id = v.split_whitespace().nth(1).unwrap_or("").trim_end_matches(':');
            let module = cat.dsls.iter().find(|d| d.id == id).map(|d| d.module_id.as_str()).unwrap_or(id);
            !(v.contains("not registered") && !enabled(module))
        })
        .collect();
    assert!(violations.is_empty(), "conformance violations:\n{}", violations.join("\n"));
}

fn run(module: &str, instruction: Value) -> buhera_registry::ActResult {
    let (mut modules, _) = federation(Options::default());
    modules.dispatch(module, instruction, 1).expect("module registered")
}

#[cfg(feature = "ndombolo")]
#[test]
fn ndombolo_demo_scores_the_manuscript_proposition() {
    let r = run("ndombolo", json!("demo"));
    assert!(r.ok);
    let d = r.output_delta.unwrap();
    assert_eq!(d["kind"], "ndombolo_result");
    assert!(d["stopped_at"].is_null());
    // Two of three readings clear 0.7 → Strong is supported at 0.9.
    let m = &d["propositions"][0]["motions"][0];
    assert_eq!(m["name"], "Strong");
    assert!((m["score"].as_f64().unwrap() - 0.9).abs() < 1e-12);
    assert!(r.residue > 0.0, "residue counts deposited events");
}

#[cfg(feature = "ndombolo")]
#[test]
fn ndombolo_stops_at_a_failing_cell_and_reports_it() {
    let r = run("ndombolo", json!("item x = 1\n// ---\nprint(undefined_name)\n// ---\nitem y = 2"));
    assert!(r.ok, "a stop is a normal, reported outcome");
    let d = r.output_delta.unwrap();
    assert_eq!(d["stopped_at"], 1);
    assert_eq!(d["cells"][1]["error"]["phase"], "run");
    assert_eq!(d["cells"].as_array().unwrap().len(), 2);
}

#[cfg(feature = "ndombolo")]
#[test]
fn ndombolo_ndo_documents_get_output_blocks_spliced() {
    let doc = "# t\n\n```turbulance\nitem a = 3\nprint(a + 1)\n```\n";
    let r = run("ndombolo", json!({ "kind": "ndo", "document": doc }));
    let out = r.output_delta.unwrap()["document"].as_str().unwrap().to_string();
    assert!(out.contains("```output\n4\n\na = 3\n```"), "{out}");
}

#[cfg(feature = "windtunnel")]
#[test]
fn windtunnel_evaluates_assertions_against_a_supplied_metric() {
    let script = "scope S:\n    include \"src/**\"\n\nassert:\n    regime >= Coherent\n    r_est >= 0.75\n    no holonomy_violations\n";
    let r = run(
        "windtunnel",
        json!({ "kind": "evaluate", "script": script, "metric": { "regime": "Coherent", "r_est": 0.81 } }),
    );
    let d = r.output_delta.unwrap();
    assert_eq!(d["verdict"], "incomplete", "holonomy was not measured → skipped");
    assert_eq!(d["passed"], 2);
    assert_eq!(d["skipped"], 1);
    assert!((r.residue - 1.0 / 3.0).abs() < 1e-12);
}

#[cfg(feature = "windtunnel")]
#[test]
fn windtunnel_parse_errors_are_collected_with_lines() {
    let r = run("windtunnel", json!("assert:\n    r_est >> 3\n"));
    assert!(!r.ok);
    assert_eq!(r.error.as_deref(), Some("parse error"));
    assert_eq!(r.residue, 1.0);
}

#[cfg(feature = "tracker")]
#[test]
fn tracker_character_of_a_small_index() {
    let sym = |name: &str, file: &str, snippet: &str| json!({ "name": name, "kind": "fn", "file": file, "line": 1, "snippet": snippet });
    let index = json!({ "root": "/r", "symbols": [
        sym("alpha_parse", "src/a.rs", "calls beta_emit"),
        sym("beta_emit", "src/b.rs", "calls alpha_parse"),
        sym("gamma_note", "docs/c.md", "prose"),
    ]});
    let r = run("tracker", json!({ "kind": "character", "index": index }));
    let d = r.output_delta.unwrap();
    assert_eq!(d["kind"], "repo_character");
    assert_eq!(d["blocks"], 3);
    assert_eq!(d["fragments"], 2, "docs/ is disconnected from src/");
    assert!(d["chi"].as_f64().unwrap() > 0.0, "χ ≥ β on the connected core");
    assert_eq!(r.residue, 0.0);
}

#[cfg(feature = "sbs-core")]
#[test]
fn sbs_core_reproduces_the_upstream_demo_numbers() {
    let r = run("sbs-core", json!("demo"));
    let d = r.output_delta.unwrap();
    // `sbs demo --perturb` in hegel: R=0.5919 V=0.1175.
    assert!((d["coherence"].as_f64().unwrap() - 0.5919).abs() < 5e-5);
    assert!((d["visibility"].as_f64().unwrap() - 0.1175).abs() < 5e-5);
    assert!((r.residue - (1.0 - 0.1175)).abs() < 5e-5);
}

#[cfg(feature = "sbs-core")]
#[test]
fn sbs_core_refuses_out_of_range_indices_instead_of_panicking() {
    let r = run("sbs-core", json!({ "kind": "observe", "circuit": { "nodes": [{ "name": "A" }], "edges": [{ "src": 0, "dst": 5 }] } }));
    assert!(!r.ok);
    assert_eq!(r.error.as_deref(), Some("invalid circuit"));
}

#[cfg(feature = "vahera")]
#[test]
fn vahera_kernel_state_persists_between_acts() {
    let (mut m, _) = federation(Options::default());
    m.dispatch("vahera", json!("memory store \"note\" = \"the deadline is Friday\""), 1).unwrap();
    let r = m.dispatch("vahera", json!("memory list"), 1).unwrap();
    assert!(serde_json::to_string(&r.output_delta).unwrap().contains("note"));
}

#[cfg(feature = "zangalewa")]
#[test]
fn zangalewa_validates_through_upstreams_own_registry_without_network() {
    let r = run("zangalewa-dsl", json!({ "kind": "validate", "dslId": "vahera", "source": "memory list" }));
    assert_eq!(r.output_delta.unwrap()["ok"], true);
    let bad = run("zangalewa-dsl", json!({ "kind": "validate", "dslId": "vahera", "source": "no such statement" }));
    assert_eq!(bad.output_delta.unwrap()["ok"], false);
}
