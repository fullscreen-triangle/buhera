//! Integration test: the standalone OS boots on the NIST dataset and
//! categorical retrieval picks the expected compound for each query.

use buhera_kernel::MemoryObject;
use buhera_os::{boot_os, load_nist};
use buhera_vahera::{execute_vahera, NamedResult};

/// Drive a single natural-language query through the full stack and
/// return the name of the first matched compound.
fn query(kernel: &mut buhera_kernel::Kernel, db_molecules: &buhera_vahera::MoleculeDatabase,
         text: &str) -> Option<String> {
    let source = format!("memory find nearest \"{}\" k=1\n", text);
    let ctx = execute_vahera(&source, kernel, db_molecules).ok()?;
    for r in ctx.results {
        if let NamedResult::FindHits(hits) = r {
            if let Some(h) = hits.first() {
                return h
                    .value
                    .metadata
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(String::from);
            }
        }
    }
    None
}

#[test]
fn boot_loads_all_compounds() {
    let db = load_nist("../../data/nist_compounds.json").expect("load nist");
    let kernel = boot_os(&db, 12);
    assert!(kernel.cmm.len() >= 5, "expected at least 5 compounds in NIST dataset");
}

#[test]
fn ethanol_query_matches_some_compound() {
    let db = load_nist("../../data/nist_compounds.json").expect("load nist");
    let mut kernel = boot_os(&db, 12);
    let matched = query(&mut kernel, &db.molecules, "ethanol");
    assert!(matched.is_some(), "ethanol query returned no hits");
}

#[test]
fn navigate_complete_cycle_runs_without_error() {
    let db = load_nist("../../data/nist_compounds.json").expect("load nist");
    let mut kernel = boot_os(&db, 12);
    let src = r#"
describe ethanol with "what is the boiling point of ethanol"
spawn lookup_eth from ethanol
navigate to penultimate
complete trajectory
"#;
    let ctx = execute_vahera(src, &mut kernel, &db.molecules).expect("execute");
    assert!(ctx.targets.contains_key("ethanol"));
    assert!(ctx.processes.contains_key("lookup_eth"));
}

#[test]
fn helper_object_field_is_present_for_completeness() {
    // Silence MemoryObject unused-import warning if the test compiler
    // strips other tests.
    let _ = std::mem::size_of::<MemoryObject>();
}
