//! `windtunnel` — the wind-tunnel `.wt` assertion language (specification
//! `specs/windtunnel.md`). Wraps vendored `wt-dsl`: `parse` and `evaluate`.
//!
//! The `.wt` language is the *deterministic half of adjudication*: it decides
//! whether a measured [`MetricView`] crossed the lines a script's author drew.
//! Measurement itself (walking a source tree with tree-sitter) is the `wt`
//! binary's job; this module reaches it only through `measure`, which spawns
//! `wt check --json --backend treesitter` when `BUHERA_WT_BIN` names the
//! binary. Without it, `measure` fails honestly — it never estimates.

use std::io::Write;
use std::process::{Command, Stdio};

use buhera_registry::{
    field_str, instruction_kind, ActResult, BindingKind, Descriptor, DslEntry, DslError, Instruction, Module,
    Validation,
};
use serde_json::{json, Value};
use wt_dsl::{evaluate, parse, MetricView, Verdict};

/// Registry id.
pub const ID: &str = "windtunnel";

/// Environment variable naming the `wt` binary for `measure`.
pub const WT_BIN_ENV: &str = "BUHERA_WT_BIN";

/// Upstream's canonical script (`wt-dsl/src/parse.rs` test `SANDBOX_DEFAULT`).
pub const DEMO_SCRIPT: &str = r#"scope PaymentFlowAnalysis:
    repo "github.com/owner/repo"
    include "src/payments/**"
    include "src/ledger/**"
    exclude "**/__tests__/**"
    language typescript

analyse:
    static
    cycles through ["checkout", "ledger", "reconciler"] max_depth 5
    purpose ablate ["audit_log", "retry_middleware"]

assert:
    regime >= Coherent
    r_est >= 0.75
    no holonomy_violations
    purposeless none in ["checkout", "ledger"]

report:
    format json
    include regime_map
    include cycle_graph
    include contribution_scores
"#;

/// The module.
#[derive(Debug, Default)]
pub struct WindTunnel;

impl WindTunnel {
    /// New module value (stateless).
    pub fn new() -> Self {
        Self
    }
}

fn parse_errors(errs: &[wt_dsl::ParseError]) -> Vec<DslError> {
    errs.iter()
        .map(|e| {
            if e.line == 0 {
                DslError::msg(e.kind.to_string())
            } else {
                DslError::at(e.kind.to_string(), e.line as u32, None)
            }
        })
        .collect()
}

/// The `.wt` front end (upstream `wt_dsl::parse`, which reports every error).
pub fn validate(source: &str) -> Validation {
    match parse(source) {
        Ok(_) => Validation::valid(),
        Err(errs) => Validation::invalid(parse_errors(&errs)),
    }
}

/// DSL registry entry.
pub fn dsl() -> DslEntry {
    DslEntry { id: "wt", label: "Wind Tunnel (.wt)", extension: ".wt", module_id: ID, pack_id: "wt", validate }
}

fn verdict_residue(failed: usize, skipped: usize, total: usize) -> f64 {
    // Fraction of assertions not yet satisfied. `incomplete` (skipped) counts
    // as not done — a skipped assertion was never checked.
    (failed + skipped) as f64 / total.max(1) as f64
}

fn parse_failure(errs: &[wt_dsl::ParseError]) -> ActResult {
    let errors = parse_errors(errs);
    let mut lines = vec![format!("windtunnel: {} parse error(s)", errors.len())];
    lines.extend(errors.iter().map(|e| format!("  line {}: {}", e.line.unwrap_or(0), e.message)));
    ActResult {
        ok: false,
        output_delta: Some(json!({ "kind": "text", "lines": lines, "errors": errors })),
        residue: 1.0,
        completed: true,
        error: Some("parse error".into()),
    }
}

impl Module for WindTunnel {
    fn id(&self) -> &str {
        ID
    }

    fn describe(&self) -> Descriptor {
        Descriptor {
            id: ID.into(),
            description: "Wind Tunnel — parse .wt analysis scripts and adjudicate their assertions (regime, R_est, \
                          R_dyn, K_c, holonomy, purposelessness) against a measured metric view; optionally \
                          measure a project with the local `wt` binary."
                .into(),
            instructions: vec![
                r#"dispatch("windtunnel", "demo")"#.into(),
                r#"dispatch("windtunnel", { kind: "evaluate", script, metric: { regime: "Coherent", r_est: 0.81 } })"#
                    .into(),
                r#"dispatch("windtunnel", { kind: "measure", script, project: "/path/to/repo" })"#.into(),
            ],
            dsl: Some("wt".into()),
            binding: BindingKind::Native,
        }
    }

    fn execute(&mut self, instruction: &Instruction, _act_budget: u32) -> ActResult {
        const EXPECTED: &str = ".wt source, \"demo\", or { kind: parse|evaluate|measure, script, … }";
        let (kind, script) = match (instruction_kind(instruction), instruction.is_object()) {
            (Some("demo") | Some(""), false) => ("parse", DEMO_SCRIPT.to_string()),
            (Some(s), false) => ("parse", s.to_string()),
            (Some(k @ ("parse" | "evaluate" | "measure")), true) => match field_str(instruction, "script") {
                Some(s) => (k, s.to_string()),
                None => return ActResult::invalid(ID, EXPECTED),
            },
            _ => return ActResult::invalid(ID, EXPECTED),
        };

        let parsed = match parse(&script) {
            Ok(s) => s,
            Err(errs) => return parse_failure(&errs),
        };

        match kind {
            "parse" => {
                let n = parsed.asserts.len();
                ActResult::done(
                    json!({ "kind": "windtunnel_script", "summary": format!("windtunnel: scope {} — {n} assertion(s)", parsed.scope.name),
                            "script": parsed }),
                    0.0,
                )
            }
            "evaluate" => {
                let metric: MetricView = match instruction.get("metric") {
                    Some(m) => match serde_json::from_value(m.clone()) {
                        Ok(v) => v,
                        Err(e) => return ActResult::fail(&[&format!("windtunnel: bad metric: {e}")], "invalid metric"),
                    },
                    None => MetricView::default(),
                };
                let report = evaluate(&parsed, &metric);
                let total = report.passed + report.failed + report.skipped;
                let mut delta = json!({
                    "kind": "windtunnel_check",
                    "summary": format!("windtunnel: {} ({} passed, {} failed, {} skipped)",
                        report.verdict.as_str(), report.passed, report.failed, report.skipped),
                    "scope": parsed.scope.name,
                    "verdict": report.verdict,
                    "passed": report.passed, "failed": report.failed, "skipped": report.skipped,
                    "results": report.results,
                    "metric": metric,
                    "measured_by": "caller",
                });
                if total == 0 {
                    delta["vacuous"] = Value::Bool(true);
                }
                let residue = verdict_residue(report.failed, report.skipped, total);
                // ok = a verdict was produced; the verdict itself is the answer.
                let _ = Verdict::Pass;
                ActResult::done(delta, residue)
            }
            _ => measure(&script, instruction),
        }
    }
}

/// Spawn `wt check <tmp.wt> <project> --json --backend treesitter`.
fn measure(script: &str, instruction: &Instruction) -> ActResult {
    let Ok(bin) = std::env::var(WT_BIN_ENV) else {
        return ActResult::fail(
            &[&format!("windtunnel measure: bridge binary not configured (set {WT_BIN_ENV})")],
            "bridge unavailable",
        );
    };
    let Some(project) = field_str(instruction, "project") else {
        return ActResult::invalid(ID, "{ kind: \"measure\", script: string, project: string }");
    };
    let tmp = std::env::temp_dir().join(format!("buhera-wt-{}.wt", std::process::id()));
    if let Err(e) = std::fs::File::create(&tmp).and_then(|mut f| f.write_all(script.as_bytes())) {
        return ActResult::fail(&[&format!("windtunnel measure: cannot write script: {e}")], "io error");
    }
    // Pinned to tree-sitter: the `auto` backend may upload source files to
    // a hosted model when tree-sitter fails (spec §Side effects).
    let out = Command::new(bin)
        .arg("check")
        .arg(&tmp)
        .arg(project)
        .args(["--json", "--backend", "treesitter"])
        .stdin(Stdio::null())
        .output();
    let _ = std::fs::remove_file(&tmp);
    let out = match out {
        Ok(o) => o,
        Err(e) => return ActResult::fail(&[&format!("windtunnel measure: cannot spawn wt: {e}")], "bridge unavailable"),
    };
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: Result<Value, _> = serde_json::from_str(stdout.trim());
    match parsed {
        Ok(mut v) if v.get("verdict").and_then(Value::as_str) != Some("parse_error") => {
            let passed = v["passed"].as_u64().unwrap_or(0) as usize;
            let failed = v["failed"].as_u64().unwrap_or(0) as usize;
            let skipped = v["skipped"].as_u64().unwrap_or(0) as usize;
            v["kind"] = json!("windtunnel_check");
            v["measured_by"] = json!("wt");
            v["summary"] = json!(format!(
                "windtunnel: {} ({passed} passed, {failed} failed, {skipped} skipped)",
                v["verdict"].as_str().unwrap_or("?")
            ));
            ActResult::done(v, verdict_residue(failed, skipped, passed + failed + skipped))
        }
        _ => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            ActResult::fail(
                &[&format!("windtunnel measure: wt exited {:?}", out.status.code()), stderr.trim()],
                "measure failed",
            )
        }
    }
}
