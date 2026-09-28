//! Parser for `.wt` scripts.
//!
//! The grammar is line-oriented and indentation-delimited: a block header ends
//! in `:` at column 0, and its directives are indented beneath it. That is the
//! shape the sandbox editor already presents, so this parses what users are
//! shown rather than a syntax invented here.
//!
//! Every error carries a 1-based line number. A parse error without a location
//! is unusable in an editor, which is the primary place these are written.

use crate::ast::*;
use crate::error::{ParseError, ParseErrorKind};

/// Parse a `.wt` script.
///
/// Returns every error found rather than only the first: a script with three
/// typos should report three, not require three round-trips.
pub fn parse(src: &str) -> Result<Script, Vec<ParseError>> {
    let mut script = Script::default();
    let mut errors: Vec<ParseError> = Vec::new();
    let mut block: Option<Block> = None;
    let mut seen_scope = false;

    for (i, raw) in src.lines().enumerate() {
        let line_no = i + 1;
        let line = strip_comment(raw);
        if line.trim().is_empty() {
            continue;
        }

        let indented = line.starts_with(char::is_whitespace);
        let text = line.trim();

        // A block header sits at column 0 and ends in ':'.
        if !indented {
            match parse_header(text) {
                Some((b, name)) => {
                    if let Block::Scope = b {
                        script.scope.name = name;
                        seen_scope = true;
                    }
                    block = Some(b);
                }
                None => errors.push(ParseError::new(
                    line_no,
                    ParseErrorKind::UnknownBlock(text.to_string()),
                )),
            }
            continue;
        }

        // Directives require a block; an indented line before any header is a
        // structural error, not something to guess a home for.
        let Some(current) = block else {
            errors.push(ParseError::new(line_no, ParseErrorKind::DirectiveOutsideBlock));
            continue;
        };

        let result = match current {
            Block::Scope   => directive_scope(&mut script.scope, text),
            Block::Analyse => directive_analyse(&mut script.analyse, text),
            Block::Assert  => directive_assert(&mut script.asserts, text),
            Block::Report  => directive_report(&mut script.report, text),
        };

        if let Err(kind) = result {
            errors.push(ParseError::new(line_no, kind));
        }
    }

    if !seen_scope {
        errors.push(ParseError::new(0, ParseErrorKind::MissingScope));
    }

    // `static` is implied by any later phase, since they consume its graph.
    let a = &mut script.analyse;
    if a.cycles.is_some() || a.purpose.is_some() || a.dynamic.is_some() {
        a.static_phase = true;
    }

    if errors.is_empty() { Ok(script) } else { Err(errors) }
}

#[derive(Clone, Copy)]
enum Block { Scope, Analyse, Assert, Report }

fn parse_header(text: &str) -> Option<(Block, String)> {
    let head = text.strip_suffix(':')?;
    let mut parts = head.split_whitespace();
    let kw = parts.next()?;
    let name = parts.collect::<Vec<_>>().join(" ");
    match kw {
        "scope"   => Some((Block::Scope, name)),
        "analyse" | "analyze" => Some((Block::Analyse, name)),
        "assert"  => Some((Block::Assert, name)),
        "report"  => Some((Block::Report, name)),
        _         => None,
    }
}

/// Strip `#` comments, but not inside a quoted string — globs legitimately
/// contain `#` and truncating one silently changes which files are analysed.
fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut in_str = false;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'"' => in_str = !in_str,
            b'#' if !in_str => return &line[..i],
            _ => {}
        }
    }
    line
}

// ── scope directives ──────────────────────────────────────────────────────────

fn directive_scope(scope: &mut Scope, text: &str) -> Result<(), ParseErrorKind> {
    let (kw, rest) = split_kw(text);
    match kw {
        "repo"     => { scope.repo = Some(string_arg(rest)?); Ok(()) }
        "include"  => { scope.include.push(string_arg(rest)?); Ok(()) }
        "exclude"  => { scope.exclude.push(string_arg(rest)?); Ok(()) }
        "language" => {
            let lang = rest.trim();
            if lang.is_empty() {
                return Err(ParseErrorKind::MissingArgument("language".into()));
            }
            scope.language = Some(lang.trim_matches('"').to_lowercase());
            Ok(())
        }
        _ => Err(ParseErrorKind::UnknownDirective { block: "scope".into(), found: kw.to_string() }),
    }
}

// ── analyse directives ────────────────────────────────────────────────────────

fn directive_analyse(a: &mut Analyse, text: &str) -> Result<(), ParseErrorKind> {
    let (kw, rest) = split_kw(text);
    match kw {
        "static" => { a.static_phase = true; Ok(()) }

        "cycles" => {
            let mut c = Cycles::default();
            let mut toks = rest.trim();
            if let Some(after) = toks.strip_prefix("through") {
                let (list, tail) = take_list(after.trim())?;
                c.through = list;
                toks = tail.trim();
            }
            if let Some(after) = toks.strip_prefix("max_depth") {
                let n = after.trim();
                c.max_depth = Some(n.parse().map_err(|_| {
                    ParseErrorKind::BadNumber(n.to_string())
                })?);
            } else if !toks.is_empty() {
                return Err(ParseErrorKind::UnexpectedTrailing(toks.to_string()));
            }
            a.cycles = Some(c);
            Ok(())
        }

        "purpose" => {
            let mut p = Purpose::default();
            let rest = rest.trim();
            if let Some(after) = rest.strip_prefix("ablate") {
                let (list, tail) = take_list(after.trim())?;
                if !tail.trim().is_empty() {
                    return Err(ParseErrorKind::UnexpectedTrailing(tail.trim().to_string()));
                }
                p.ablate = list;
            } else if !rest.is_empty() {
                return Err(ParseErrorKind::UnexpectedTrailing(rest.to_string()));
            }
            a.purpose = Some(p);
            Ok(())
        }

        "dynamic" => {
            let rest = rest.trim();
            let traces = rest
                .strip_prefix("traces")
                .ok_or(ParseErrorKind::MissingArgument("dynamic traces".into()))?;
            a.dynamic = Some(Dynamic { traces: string_arg(traces)? });
            Ok(())
        }

        _ => Err(ParseErrorKind::UnknownDirective { block: "analyse".into(), found: kw.to_string() }),
    }
}

// ── assert directives ─────────────────────────────────────────────────────────

fn directive_assert(out: &mut Vec<Assertion>, text: &str) -> Result<(), ParseErrorKind> {
    let (kw, rest) = split_kw(text);

    // `no <thing>`
    if kw == "no" {
        return match rest.trim() {
            "holonomy_violations" => { out.push(Assertion::NoHolonomyViolations); Ok(()) }
            "cycles"              => { out.push(Assertion::NoCycles); Ok(()) }
            other => Err(ParseErrorKind::UnknownAssertion(format!("no {other}"))),
        };
    }

    // `purposeless none in [...]`
    if kw == "purposeless" {
        let rest = rest.trim();
        let after_none = rest
            .strip_prefix("none")
            .ok_or_else(|| ParseErrorKind::UnknownAssertion(format!("purposeless {rest}")))?;
        let after_in = after_none.trim().strip_prefix("in").unwrap_or("").trim();
        let units = if after_in.is_empty() {
            Vec::new()
        } else {
            take_list(after_in)?.0
        };
        out.push(Assertion::PurposelessNoneIn { units });
        return Ok(());
    }

    // `<field> <op> <value>`
    let mut toks = text.split_whitespace();
    let field = toks.next().unwrap_or("");
    let op_s  = toks.next().ok_or(ParseErrorKind::MissingArgument("comparison operator".into()))?;
    let val_s = toks.next().ok_or(ParseErrorKind::MissingArgument("comparison value".into()))?;
    if let Some(extra) = toks.next() {
        return Err(ParseErrorKind::UnexpectedTrailing(extra.to_string()));
    }

    let op = CmpOp::parse(op_s).ok_or_else(|| ParseErrorKind::BadOperator(op_s.to_string()))?;

    if field == "regime" {
        let regime = canonical_regime(val_s)
            .ok_or_else(|| ParseErrorKind::UnknownRegime(val_s.to_string()))?;
        out.push(Assertion::Regime { op, regime });
        return Ok(());
    }

    let sf = ScalarField::parse(field)
        .ok_or_else(|| ParseErrorKind::UnknownAssertion(field.to_string()))?;
    let value: f64 = val_s.parse().map_err(|_| ParseErrorKind::BadNumber(val_s.to_string()))?;
    out.push(Assertion::Scalar { field: sf, op, value });
    Ok(())
}

/// Accept the regime names users actually write — the enum variant, the
/// hyphenated display form, and case variations of both.
pub fn canonical_regime(s: &str) -> Option<String> {
    let norm = s.trim().trim_matches('"').to_lowercase().replace([' ', '-', '_'], "");
    let out = match norm.as_str() {
        "turbulent"                                  => "Turbulent",
        "aperturedominated"                          => "ApertureDominated",
        "hierarchicalcascade"                        => "HierarchicalCascade",
        "coherent"                                   => "Coherent",
        "phaselocked"                                => "PhaseLocked",
        _ => return None,
    };
    Some(out.to_string())
}

// ── report directives ─────────────────────────────────────────────────────────

fn directive_report(r: &mut Report, text: &str) -> Result<(), ParseErrorKind> {
    let (kw, rest) = split_kw(text);
    match kw {
        "format" => {
            r.format = match rest.trim() {
                "json" => Format::Json,
                "text" => Format::Text,
                other  => return Err(ParseErrorKind::UnknownFormat(other.to_string())),
            };
            Ok(())
        }
        "include" => {
            let s = rest.trim();
            if s.is_empty() {
                return Err(ParseErrorKind::MissingArgument("include".into()));
            }
            r.include.push(s.trim_matches('"').to_string());
            Ok(())
        }
        _ => Err(ParseErrorKind::UnknownDirective { block: "report".into(), found: kw.to_string() }),
    }
}

// ── shared helpers ────────────────────────────────────────────────────────────

fn split_kw(text: &str) -> (&str, &str) {
    match text.find(char::is_whitespace) {
        Some(i) => (&text[..i], &text[i..]),
        None    => (text, ""),
    }
}

fn string_arg(rest: &str) -> Result<String, ParseErrorKind> {
    let s = rest.trim();
    if s.is_empty() {
        return Err(ParseErrorKind::MissingArgument("quoted string".into()));
    }
    Ok(s.trim_matches('"').to_string())
}

/// Take a leading `[a, b, c]` list, returning it and whatever follows.
fn take_list(s: &str) -> Result<(Vec<String>, &str), ParseErrorKind> {
    let s = s.trim();
    if !s.starts_with('[') {
        return Err(ParseErrorKind::ExpectedList(s.chars().take(20).collect()));
    }
    let end = s.find(']').ok_or(ParseErrorKind::UnclosedList)?;
    let inner = &s[1..end];
    let items: Vec<String> = inner
        .split(',')
        .map(|p| p.trim().trim_matches('"').to_string())
        .filter(|p| !p.is_empty())
        .collect();
    Ok((items, &s[end + 1..]))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact script the sandbox editor ships as its default. If this ever
    /// fails to parse, the web tool is showing users something the compiler
    /// rejects.
    const SANDBOX_DEFAULT: &str = r#"# Wind Tunnel DSL — directed analysis script

scope PaymentFlowAnalysis:
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

    #[test]
    fn parses_the_sandbox_default_script() {
        let s = parse(SANDBOX_DEFAULT).expect("sandbox default must parse");

        assert_eq!(s.scope.name, "PaymentFlowAnalysis");
        assert_eq!(s.scope.repo.as_deref(), Some("github.com/owner/repo"));
        assert_eq!(s.scope.include.len(), 2);
        assert_eq!(s.scope.exclude, vec!["**/__tests__/**"]);
        assert_eq!(s.scope.language.as_deref(), Some("typescript"));

        assert!(s.analyse.static_phase);
        let c = s.analyse.cycles.as_ref().unwrap();
        assert_eq!(c.through, vec!["checkout", "ledger", "reconciler"]);
        assert_eq!(c.max_depth, Some(5));
        assert_eq!(
            s.analyse.purpose.as_ref().unwrap().ablate,
            vec!["audit_log", "retry_middleware"]
        );

        assert_eq!(s.asserts.len(), 4);
        assert_eq!(s.report.format, Format::Json);
        assert_eq!(s.report.include.len(), 3);
    }

    #[test]
    fn later_phases_imply_static() {
        let src = "scope S:\nanalyse:\n    purpose ablate [\"a\"]\n";
        let s = parse(src).unwrap();
        assert!(s.analyse.static_phase, "purpose consumes the static graph");
    }

    #[test]
    fn regime_names_accept_display_and_variant_forms() {
        for form in ["Coherent", "coherent", "Phase-locked", "phase_locked", "PhaseLocked"] {
            assert!(canonical_regime(form).is_some(), "{form} should be accepted");
        }
        assert!(canonical_regime("Sideways").is_none());
    }

    #[test]
    fn reports_every_error_not_just_the_first() {
        let src = "scope S:\nassert:\n    r_est >= abc\n    regime >= Sideways\n";
        let errs = parse(src).unwrap_err();
        assert_eq!(errs.len(), 2, "both bad lines should be reported");
        assert_eq!(errs[0].line, 3);
        assert_eq!(errs[1].line, 4);
    }

    #[test]
    fn missing_scope_is_an_error() {
        let errs = parse("analyse:\n    static\n").unwrap_err();
        assert!(errs.iter().any(|e| matches!(e.kind, ParseErrorKind::MissingScope)));
    }

    #[test]
    fn hash_inside_a_quoted_glob_is_not_a_comment() {
        let s = parse("scope S:\n    include \"src/#special/**\"\n").unwrap();
        assert_eq!(s.scope.include, vec!["src/#special/**"]);
    }

    #[test]
    fn directive_before_any_block_is_rejected() {
        let errs = parse("    static\nscope S:\n").unwrap_err();
        assert!(errs.iter().any(|e| matches!(e.kind, ParseErrorKind::DirectiveOutsideBlock)));
    }

    #[test]
    fn unknown_directive_names_its_block() {
        let errs = parse("scope S:\n    frobnicate \"x\"\n").unwrap_err();
        match &errs[0].kind {
            ParseErrorKind::UnknownDirective { block, found } => {
                assert_eq!(block, "scope");
                assert_eq!(found, "frobnicate");
            }
            other => panic!("expected UnknownDirective, got {other:?}"),
        }
    }

    #[test]
    fn purposeless_without_a_list_means_no_unit_anywhere() {
        let s = parse("scope S:\nassert:\n    purposeless none\n").unwrap();
        assert_eq!(s.asserts, vec![Assertion::PurposelessNoneIn { units: vec![] }]);
    }

    #[test]
    fn analyze_spelling_is_accepted() {
        let s = parse("scope S:\nanalyze:\n    static\n").unwrap();
        assert!(s.analyse.static_phase);
    }
}
