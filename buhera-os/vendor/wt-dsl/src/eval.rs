//! Assertion evaluation — where a measured metric becomes a verdict.
//!
//! This is the deterministic half of adjudication. It answers "did the numbers
//! cross the lines the author drew", and nothing else. It has no access to
//! intent, so it never claims to judge whether a failure *matters*; see
//! `docs/testing-pipelines-deployment.md` §4 for the half that does.
//!
//! Two properties this deliberately has:
//!
//! * **Every check runs.** No short-circuit on first failure — a run that
//!   reports one of four failures forces four round-trips.
//! * **An assertion whose input is absent is `Skipped`, never `Passed`.**
//!   Asserting `r_dyn >= 0.8` with no traces must not report success; that
//!   would be a green build certifying a measurement that never happened.

use serde::{Deserialize, Serialize};

use crate::ast::{Assertion, ScalarField, Script};

// ── Metric view ───────────────────────────────────────────────────────────────

/// The subset of `WindTunnelMetric` assertions can reach.
///
/// This crate does not depend on `wt-report`: the DSL is also compiled to
/// WASM for the browser, where no analysis crates are present. The CLI fills
/// this in from the real metric; the shape is the contract between them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricView {
    pub regime:              Option<String>,
    pub r_est:               Option<f64>,
    pub r_dyn:               Option<f64>,
    pub k_c:                 Option<f64>,
    pub s_flat_est:          Option<f64>,
    /// `None` = the dynamic phase did not run. `Some(0)` = it ran and found none.
    pub holonomy_violations: Option<usize>,
    pub cycle_candidates:    Option<usize>,
    /// `None` = the purpose phase did not run.
    pub purposeless:         Option<Vec<String>>,
}

/// Rank regimes for `>=` comparisons. `wt_static::Regime` is `Eq` but not
/// `Ord`, and ordering is a DSL concern rather than a property of the metric.
fn regime_rank(name: &str) -> Option<u8> {
    let norm = name.trim().to_lowercase().replace([' ', '-', '_'], "");
    Some(match norm.as_str() {
        "turbulent"           => 0,
        "aperturedominated"   => 1,
        "hierarchicalcascade" => 2,
        "coherent"            => 3,
        "phaselocked"         => 4,
        _ => return None,
    })
}

// ── Outcome ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome { Passed, Failed, Skipped }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertionResult {
    /// The assertion as written, for display.
    pub source:  String,
    pub outcome: Outcome,
    /// Why it failed, or which phase was missing when skipped.
    pub detail:  String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Every assertion passed and none were skipped.
    Pass,
    /// At least one assertion failed.
    Fail,
    /// Nothing failed, but some assertion could not be evaluated.
    Incomplete,
}

impl Verdict {
    /// Process exit code. `Incomplete` is deliberately non-zero: a pipeline
    /// that treats "could not check" as "checked and fine" has no gate.
    pub fn exit_code(&self) -> i32 {
        match self {
            Verdict::Pass       => 0,
            Verdict::Fail       => 1,
            Verdict::Incomplete => 3,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Pass       => "pass",
            Verdict::Fail       => "fail",
            Verdict::Incomplete => "incomplete",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub verdict: Verdict,
    pub results: Vec<AssertionResult>,
    pub passed:  usize,
    pub failed:  usize,
    pub skipped: usize,
}

// ── Evaluation ────────────────────────────────────────────────────────────────

pub fn evaluate(script: &Script, m: &MetricView) -> EvalReport {
    let results: Vec<AssertionResult> =
        script.asserts.iter().map(|a| eval_one(a, m)).collect();

    let passed  = results.iter().filter(|r| r.outcome == Outcome::Passed).count();
    let failed  = results.iter().filter(|r| r.outcome == Outcome::Failed).count();
    let skipped = results.iter().filter(|r| r.outcome == Outcome::Skipped).count();

    let verdict = if failed > 0 {
        Verdict::Fail
    } else if skipped > 0 {
        Verdict::Incomplete
    } else {
        Verdict::Pass
    };

    EvalReport { verdict, results, passed, failed, skipped }
}

fn eval_one(a: &Assertion, m: &MetricView) -> AssertionResult {
    let source = render(a);

    macro_rules! skip {
        ($why:expr) => {
            return AssertionResult {
                source,
                outcome: Outcome::Skipped,
                detail: $why.to_string(),
            }
        };
    }
    macro_rules! decide {
        ($ok:expr, $detail:expr) => {
            return AssertionResult {
                source,
                outcome: if $ok { Outcome::Passed } else { Outcome::Failed },
                detail: $detail,
            }
        };
    }

    match a {
        Assertion::Regime { op, regime } => {
            let Some(actual) = m.regime.as_deref() else {
                skip!("no regime — the static phase did not run");
            };
            let (Some(lhs), Some(rhs)) = (regime_rank(actual), regime_rank(regime)) else {
                skip!(format!("cannot rank regime `{actual}`"));
            };
            decide!(
                op.apply(lhs as f64, rhs as f64),
                format!("regime is {actual}, required {} {regime}", op.as_str())
            );
        }

        Assertion::Scalar { field, op, value } => {
            let actual = match field {
                ScalarField::REst     => m.r_est,
                ScalarField::RDyn     => m.r_dyn,
                ScalarField::KC       => m.k_c,
                ScalarField::SFlatEst => m.s_flat_est,
            };
            let Some(actual) = actual else {
                skip!(match field {
                    ScalarField::RDyn | ScalarField::SFlatEst =>
                        format!("no {} — the dynamic phase did not run", field.as_str()),
                    _ => format!("no {} in the metric", field.as_str()),
                });
            };
            decide!(
                op.apply(actual, *value),
                format!("{} is {actual:.4}, required {} {value}", field.as_str(), op.as_str())
            );
        }

        Assertion::NoHolonomyViolations => {
            let Some(n) = m.holonomy_violations else {
                skip!("no holonomy data — the dynamic phase did not run");
            };
            decide!(n == 0, format!("{n} holonomy violation(s)"));
        }

        Assertion::NoCycles => {
            let Some(n) = m.cycle_candidates else {
                skip!("no cycle data — the static phase did not run");
            };
            decide!(n == 0, format!("{n} cycle candidate(s)"));
        }

        Assertion::PurposelessNoneIn { units } => {
            let Some(purposeless) = m.purposeless.as_ref() else {
                skip!("no contribution scores — the purpose phase did not run");
            };
            // An empty list means "no unit anywhere may be purposeless".
            let offenders: Vec<&String> = if units.is_empty() {
                purposeless.iter().collect()
            } else {
                purposeless.iter().filter(|p| units.contains(p)).collect()
            };
            decide!(
                offenders.is_empty(),
                format!("purposeless: {offenders:?}")
            );
        }
    }
}

/// Render an assertion back to its source form, so a result can be shown next
/// to what the author wrote rather than a paraphrase.
pub fn render(a: &Assertion) -> String {
    match a {
        Assertion::Regime { op, regime } =>
            format!("regime {} {regime}", op.as_str()),
        Assertion::Scalar { field, op, value } =>
            format!("{} {} {value}", field.as_str(), op.as_str()),
        Assertion::NoHolonomyViolations =>
            "no holonomy_violations".to_string(),
        Assertion::NoCycles =>
            "no cycles".to_string(),
        Assertion::PurposelessNoneIn { units } if units.is_empty() =>
            "purposeless none".to_string(),
        Assertion::PurposelessNoneIn { units } =>
            format!("purposeless none in {units:?}"),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    fn script(asserts: &str) -> Script {
        parse(&format!("scope S:\nassert:\n{asserts}")).expect("test script must parse")
    }

    fn static_only() -> MetricView {
        MetricView {
            regime:           Some("HierarchicalCascade".into()),
            r_est:            Some(0.5333),
            k_c:              Some(0.1504),
            cycle_candidates: Some(0),
            ..Default::default()
        }
    }

    #[test]
    fn regime_comparison_is_ordinal_not_lexical() {
        // "HierarchicalCascade" > "Coherent" as strings, but ranks below it.
        let r = evaluate(&script("    regime >= Coherent\n"), &static_only());
        assert_eq!(r.verdict, Verdict::Fail);
    }

    #[test]
    fn missing_phase_is_skipped_never_passed() {
        // r_dyn was never measured; this must not report success.
        let r = evaluate(&script("    r_dyn >= 0.8\n"), &static_only());
        assert_eq!(r.results[0].outcome, Outcome::Skipped);
        assert_eq!(r.verdict, Verdict::Incomplete);
        assert_ne!(r.verdict.exit_code(), 0, "incomplete must not exit 0");
    }

    #[test]
    fn zero_violations_differs_from_absent_violations() {
        let mut ran = static_only();
        ran.holonomy_violations = Some(0);
        let r = evaluate(&script("    no holonomy_violations\n"), &ran);
        assert_eq!(r.results[0].outcome, Outcome::Passed);

        // Same assertion, dynamic phase never run.
        let r = evaluate(&script("    no holonomy_violations\n"), &static_only());
        assert_eq!(r.results[0].outcome, Outcome::Skipped);
    }

    #[test]
    fn every_assertion_is_evaluated_not_short_circuited() {
        let s = script("    regime >= Coherent\n    r_est >= 0.75\n    no cycles\n");
        let r = evaluate(&s, &static_only());
        assert_eq!(r.results.len(), 3);
        assert_eq!(r.failed, 2);
        assert_eq!(r.passed, 1, "`no cycles` holds and must still be reported");
    }

    #[test]
    fn purposeless_empty_list_means_any_unit() {
        let mut m = static_only();
        m.purposeless = Some(vec!["audit_log".into()]);
        let r = evaluate(&script("    purposeless none\n"), &m);
        assert_eq!(r.results[0].outcome, Outcome::Failed);
    }

    #[test]
    fn purposeless_scoped_list_ignores_units_outside_it() {
        let mut m = static_only();
        m.purposeless = Some(vec!["audit_log".into()]);
        let r = evaluate(&script("    purposeless none in [\"checkout\"]\n"), &m);
        assert_eq!(r.results[0].outcome, Outcome::Passed);
    }

    #[test]
    fn failure_detail_names_the_actual_value() {
        let r = evaluate(&script("    r_est >= 0.75\n"), &static_only());
        assert!(r.results[0].detail.contains("0.5333"), "got: {}", r.results[0].detail);
    }

    #[test]
    fn all_pass_gives_exit_zero() {
        let s = script("    regime >= HierarchicalCascade\n    r_est >= 0.5\n");
        let r = evaluate(&s, &static_only());
        assert_eq!(r.verdict, Verdict::Pass);
        assert_eq!(r.verdict.exit_code(), 0);
    }

    #[test]
    fn a_script_with_no_assertions_passes_vacuously() {
        let s = parse("scope S:\nanalyse:\n    static\n").unwrap();
        assert_eq!(evaluate(&s, &static_only()).verdict, Verdict::Pass);
    }
}
