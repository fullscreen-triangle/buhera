//! `tracker` — the repo character invariant χ from bloodhound's
//! repo-federation tracker (specification `specs/tracker.md`). Wraps the
//! vendored `chi.rs` verbatim.
//!
//! Only the read-only half of the tracker is exposed. Its mutating verbs —
//! `add`/`drift` (advance the monotone act counter), `sync` (pushes to
//! external remotes), `profile git-setup --apply` (rewrites global git
//! config) — are deliberately not dispatchable (spec 07, B5).

use buhera_registry::{field_str, instruction_kind, ActResult, BindingKind, Descriptor, Instruction, Module};
use serde_json::{json, Value};
use tracker_chi::chi;
use tracker_chi::purpose::Index;

/// Registry id.
pub const ID: &str = "tracker";

/// The module.
#[derive(Debug, Default)]
pub struct Tracker {
    /// Whether filesystem-reading operations (`character_at`, `list`) are
    /// enabled. Off in the wasm build, which has no filesystem.
    pub filesystem: bool,
}

impl Tracker {
    /// A tracker that may read `.purpose/index.json` and `.tracker/federation.json`.
    pub fn with_filesystem() -> Self {
        Self { filesystem: true }
    }
}

fn character_delta(index: &Index, repo: Option<&str>) -> ActResult {
    let c = chi::compute(index);
    let summary = format!(
        "tracker: χ = {:.3} (core {} of {} blocks; {} fragment(s))",
        c.chi, c.core_blocks, c.blocks, c.fragments
    );
    ActResult::done(
        json!({
            "kind": "repo_character",
            "summary": summary,
            "repo": repo,
            "chi": c.chi,
            "blocks": c.blocks,
            "core_blocks": c.core_blocks,
            "fragments": c.fragments,
            "cut_side": c.cut_side,
            "salient": c.salient.iter().map(|(f, d)| json!({ "block": f, "degree": d })).collect::<Vec<_>>(),
            "beta": chi::BETA,
        }),
        // χ is a conserved invariant, not a distance: nothing remains to do.
        0.0,
    )
}

fn parse_index(v: &Value) -> Result<Index, String> {
    serde_json::from_value(v.clone()).map_err(|e| format!("index is not a purpose index ({{root, symbols}}): {e}"))
}

impl Module for Tracker {
    fn id(&self) -> &str {
        ID
    }

    fn describe(&self) -> Descriptor {
        Descriptor {
            id: ID.into(),
            description: "tracker — the repo character invariant χ: the Stoer–Wagner minimum cut of a repo's \
                          `purpose` self-graph (largest component), with the load-bearing blocks and fragment \
                          count. Read-only; the tracker's mutating verbs are not dispatchable."
                .into(),
            instructions: vec![
                r#"dispatch("tracker", { kind: "character", index: { root, symbols: [...] } })"#.into(),
                r#"dispatch("tracker", { kind: "character_at", path: "/path/to/repo" })"#.into(),
                r#"dispatch("tracker", { kind: "list", root: "/path/to/federation" })"#.into(),
            ],
            dsl: None,
            binding: BindingKind::Native,
        }
    }

    fn execute(&mut self, instruction: &Instruction, _act_budget: u32) -> ActResult {
        const EXPECTED: &str = "{ kind: character|character_at|list, … }";
        match instruction_kind(instruction) {
            Some("character") => {
                let Some(v) = instruction.get("index") else {
                    return ActResult::invalid(ID, "{ kind: \"character\", index: { root, symbols } }");
                };
                match parse_index(v) {
                    Ok(index) => character_delta(&index, field_str(instruction, "repo")),
                    Err(e) => ActResult::fail(&[&format!("tracker: {e}")], "invalid index"),
                }
            }
            Some("character_at") | Some("list") if !self.filesystem => ActResult::fail(
                &["tracker: this host has no filesystem access; pass { kind: \"character\", index }"],
                "unavailable on this host",
            ),
            Some("character_at") => {
                let Some(path) = field_str(instruction, "path") else {
                    return ActResult::invalid(ID, "{ kind: \"character_at\", path: string }");
                };
                let file = std::path::Path::new(path).join(".purpose").join("index.json");
                let text = match std::fs::read(&file) {
                    Ok(b) => String::from_utf8_lossy(&b).trim_start_matches('\u{feff}').to_string(),
                    Err(e) => {
                        return ActResult::fail(
                            &[&format!("tracker: cannot read {} ({e}); run `purpose index` in the repo", file.display())],
                            "index unreadable",
                        )
                    }
                };
                match serde_json::from_str::<Value>(&text).map_err(|e| e.to_string()).and_then(|v| parse_index(&v)) {
                    Ok(index) => character_delta(&index, Some(path)),
                    Err(e) => ActResult::fail(&[&format!("tracker: {e}")], "index malformed"),
                }
            }
            Some("list") => {
                let Some(root) = field_str(instruction, "root") else {
                    return ActResult::invalid(ID, "{ kind: \"list\", root: string }");
                };
                let file = std::path::Path::new(root).join(".tracker").join("federation.json");
                match std::fs::read_to_string(&file).map_err(|e| e.to_string()).and_then(|t| {
                    serde_json::from_str::<Value>(&t).map_err(|e| e.to_string())
                }) {
                    Ok(fed) => {
                        let repos = fed.get("repos").cloned().unwrap_or(Value::Array(vec![]));
                        let n = repos.as_array().map_or(0, Vec::len);
                        ActResult::done(
                            json!({ "kind": "repo_federation", "summary": format!("tracker: {n} tracked repo(s)"), "repos": repos }),
                            0.0,
                        )
                    }
                    Err(e) => ActResult::fail(&[&format!("tracker: cannot read {}: {e}", file.display())], "federation unreadable"),
                }
            }
            _ => ActResult::invalid(ID, EXPECTED),
        }
    }
}
