//! `heihachi` — the two languages of the heihachi micro-kernel (specification
//! `specs/heihachi.md`): `mishima` (`.mma`) computes over the accumulated
//! record, `sangoma` (`.sgn`) constructs material against declared targets.
//! Wraps the vendored front ends: `parse` then `check`.
//!
//! A check never throws: a refusal is a [`Diagnostic`] with a rule, a message
//! and a remedy, anchored to a line. Running a program (driving the record
//! graph) is orchestrated upstream inside the daemon's HTTP handler, not in a
//! library function, so this module checks and does not run (U-hei-1).

use buhera_registry::{
    field_str, ActResult, BindingKind, Descriptor, DslEntry, DslError, Instruction, Module, Validation,
};
use heihachi::lang::{self, mishima, sangoma, CheckResult, Diagnostic, Severity};
use serde_json::{json, Value};

/// Registry id.
pub const ID: &str = "heihachi";

/// The finest resolution the render path delivers; the CLI and server default.
pub const DEFAULT_BACKEND_RESOLUTION: f64 = 0.001;
/// The composite power a sangoma construct must reach; hard-coded upstream.
pub const DEFAULT_REQUIRED_POWER: f64 = 0.8;

/// Upstream `examples/recall.mma`.
pub const DEMO_MISHIMA: &str = include_str!("../../../vendor/heihachi/examples/recall.mma");
/// Upstream `examples/reese.sgn`.
pub const DEMO_SANGOMA: &str = include_str!("../../../vendor/heihachi/examples/reese.sgn");

/// The two languages this module owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// `.mma` — seeks over the record.
    Mishima,
    /// `.sgn` — constructs against targets.
    Sangoma,
}

impl Language {
    fn id(self) -> &'static str {
        match self {
            Language::Mishima => "mishima",
            Language::Sangoma => "sangoma",
        }
    }
}

/// One seek (mishima) or construct (sangoma), with its ladder's composite power.
fn composites_mishima(p: &mishima::Program) -> Vec<Value> {
    p.seeks
        .iter()
        .map(|s| {
            json!({
                "kind": "seek", "name": s.subject, "line": s.line,
                "rungs": s.ladder.iter().map(|r| json!({ "name": r.name, "power": r.power })).collect::<Vec<_>>(),
                "composite_power": lang::composite_power(s.ladder.iter().map(|r| r.power)),
            })
        })
        .collect()
}

fn composites_sangoma(p: &sangoma::Program) -> Vec<Value> {
    p.constructs
        .iter()
        .map(|c| {
            json!({
                "kind": "construct", "name": c.name, "line": c.line,
                "rungs": c.ladder.iter().map(|r| json!({ "name": r.name, "power": r.power })).collect::<Vec<_>>(),
                "composite_power": lang::composite_power(c.ladder.iter().map(|r| r.power)),
            })
        })
        .collect()
}

/// Parse + check. A parse failure is a single-diagnostic refusal.
pub fn check(language: Language, source: &str, backend_resolution: f64, required_power: f64) -> (CheckResult, Vec<Value>) {
    let parsed = match language {
        Language::Mishima => mishima::parse(source).map(|p| (mishima::check(&p, backend_resolution), composites_mishima(&p))),
        Language::Sangoma => sangoma::parse(source)
            .map(|p| (sangoma::check(&p, required_power, backend_resolution), composites_sangoma(&p))),
    };
    match parsed {
        Ok(r) => r,
        Err(d) => (CheckResult::from_diagnostics(vec![d], Vec::new()), Vec::new()),
    }
}

fn dsl_errors(r: &CheckResult) -> Vec<DslError> {
    r.diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d: &Diagnostic| DslError::at(format!("[{}] {} — {}", d.rule, d.message, d.remedy), d.line as u32, Some(d.column as u32)))
        .collect()
}

fn validate_with(language: Language, source: &str) -> Validation {
    let (r, _) = check(language, source, DEFAULT_BACKEND_RESOLUTION, DEFAULT_REQUIRED_POWER);
    if r.accepted {
        Validation::valid()
    } else {
        Validation::invalid(dsl_errors(&r))
    }
}

/// The mishima front end, at the CLI's default backend resolution.
pub fn validate_mishima(source: &str) -> Validation {
    validate_with(Language::Mishima, source)
}

/// The sangoma front end, at the CLI's default required power and resolution.
pub fn validate_sangoma(source: &str) -> Validation {
    validate_with(Language::Sangoma, source)
}

/// DSL registry entries (both languages route to this module).
pub fn dsls() -> [DslEntry; 2] {
    [
        DslEntry { id: "mishima", label: "mishima", extension: ".mma", module_id: ID, pack_id: "mishima", validate: validate_mishima },
        DslEntry { id: "sangoma", label: "sangoma", extension: ".sgn", module_id: ID, pack_id: "sangoma", validate: validate_sangoma },
    ]
}

/// The module (stateless).
#[derive(Debug, Default)]
pub struct Heihachi;

impl Heihachi {
    /// New module value.
    pub fn new() -> Self {
        Self
    }
}

fn number(instruction: &Instruction, key: &str, default: f64) -> Result<f64, ()> {
    match instruction.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v.as_f64().filter(|x| x.is_finite()).ok_or(()),
    }
}

impl Module for Heihachi {
    fn id(&self) -> &str {
        ID
    }

    fn describe(&self) -> Descriptor {
        Descriptor {
            id: ID.into(),
            description: "heihachi — check mishima (.mma) seeks and sangoma (.sgn) constructs: floor declared and \
                          negotiable with the render path, every seek bounded by a `not` clause, ladders coherent \
                          (≥ 3 rungs to closure) and not saturated; report each ladder's composite power. Refusals \
                          carry a rule and a remedy."
                .into(),
            instructions: vec![
                r#"dispatch("heihachi", "demo")"#.into(),
                r#"dispatch("heihachi", { kind: "check", language: "sangoma", source, required_power: 0.8 })"#.into(),
                r#"dispatch("heihachi", "<mishima source>")"#.into(),
            ],
            dsl: Some("mishima".into()),
            binding: BindingKind::Native,
        }
    }

    fn execute(&mut self, instruction: &Instruction, _act_budget: u32) -> ActResult {
        const EXPECTED: &str =
            "mishima source, \"demo\", or { kind: \"check\", language: \"mishima\" | \"sangoma\", source, backend_resolution?, required_power? }";
        let (language, source, backend, power) = match instruction {
            Value::String(s) if s == "demo" => (Language::Sangoma, DEMO_SANGOMA.to_string(), DEFAULT_BACKEND_RESOLUTION, DEFAULT_REQUIRED_POWER),
            Value::String(s) => (Language::Mishima, s.clone(), DEFAULT_BACKEND_RESOLUTION, DEFAULT_REQUIRED_POWER),
            Value::Object(_) => {
                if !matches!(field_str(instruction, "kind"), None | Some("check")) {
                    return ActResult::invalid(ID, EXPECTED);
                }
                let language = match field_str(instruction, "language") {
                    None | Some("mishima") => Language::Mishima,
                    Some("sangoma") => Language::Sangoma,
                    Some(_) => return ActResult::invalid(ID, EXPECTED),
                };
                let Some(source) = field_str(instruction, "source") else { return ActResult::invalid(ID, EXPECTED) };
                let (Ok(backend), Ok(power)) = (
                    number(instruction, "backend_resolution", DEFAULT_BACKEND_RESOLUTION),
                    number(instruction, "required_power", DEFAULT_REQUIRED_POWER),
                ) else {
                    return ActResult::invalid(ID, EXPECTED);
                };
                (language, source.to_string(), backend, power)
            }
            _ => return ActResult::invalid(ID, EXPECTED),
        };

        let (result, composites) = check(language, &source, backend, power);
        let errors = result.diagnostics.iter().filter(|d| d.severity == Severity::Error).count();
        let warnings = result.diagnostics.len() - errors;
        let delta = json!({
            "kind": "heihachi_check",
            "language": language.id(),
            "accepted": result.accepted,
            "summary": format!("{}: {} ({errors} error(s), {warnings} warning(s))", language.id(),
                               if result.accepted { "accepted" } else { "refused" }),
            "diagnostics": result.diagnostics,
            "declarations": result.declarations,
            "ladders": composites,
            "backend_resolution": backend,
            "required_power": power,
        });
        if result.accepted {
            ActResult::done(delta, 0.0)
        } else {
            let first = result.diagnostics.iter().find(|d| d.severity == Severity::Error).map(|d| d.message.clone());
            ActResult {
                ok: false,
                output_delta: Some(delta),
                // Check-only residue: the refusals still to fix.
                residue: errors as f64,
                completed: true,
                error: Some(first.unwrap_or_else(|| "refused".into())),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_examples_check_clean() {
        assert!(validate_mishima(DEMO_MISHIMA).ok);
        assert!(validate_sangoma(DEMO_SANGOMA).ok);
    }

    #[test]
    fn the_demo_reports_reeses_composite_power() {
        let r = Heihachi::new().execute(&json!("demo"), 1);
        assert!(r.ok, "{:?}", r.error);
        let d = r.output_delta.unwrap();
        let cp = d["ladders"][0]["composite_power"].as_f64().unwrap();
        // The example's own comment: 1 - (0.60)(0.65)(0.45) = 0.8245.
        assert!((cp - 0.8245).abs() < 1e-12, "{cp}");
    }

    #[test]
    fn refusals_carry_rule_line_and_remedy() {
        let v = validate_mishima("floor 0.02\nseek x\n  toward { region(y) }\n  via { rung a at 0.4 >> rung b at 0.4 >> rung c at 0.4 }\n  until closure\n  yield r\n");
        assert!(!v.ok);
        assert!(v.errors[0].message.contains("rule:mandatory-not"), "{}", v.errors[0].message);
        assert!(v.errors[0].line.is_some());
        let floor0 = Heihachi::new().execute(&json!({ "language": "mishima", "source": "floor 0.0\n" }), 1);
        assert!(!floor0.ok);
        // A finer floor than the render path resolves is refused by negotiation.
        let fine = Heihachi::new().execute(
            &json!({ "kind": "check", "language": "sangoma", "source": DEMO_SANGOMA, "backend_resolution": 0.05 }),
            1,
        );
        assert!(!fine.ok);
        assert!(fine.output_delta.unwrap()["diagnostics"].to_string().contains("floor-negotiation"));
    }
}
