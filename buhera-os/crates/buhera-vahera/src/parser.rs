//! vaHera parser.
//!
//! Line-oriented: blank lines and `# ...` comments are ignored; every
//! other non-empty line is parsed into exactly one [`Stmt`].

use buhera_substrate::SCoord;
use regex::Regex;
use thiserror::Error;

use crate::ast::{Stmt, StmtKind};

/// Parse error.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ParseError {
    /// A line did not match any known statement.
    #[error("syntax error at line {line}: {message}")]
    Syntax {
        /// 1-indexed source line.
        line: usize,
        /// Reason.
        message: String,
    },
}

/// Parse `source` into a vector of [`Stmt`].
pub fn parse_vahera(source: &str) -> Result<Vec<Stmt>, ParseError> {
    let s_coord = Regex::new(r"S\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)")
        .expect("static regex");
    let describe = Regex::new(r#"^describe\s+(\S+)\s+with\s+"([^"]*)"$"#).expect("static regex");
    let spawn = Regex::new(r#"^spawn\s+(\S+)\s+from\s+(\S+)$"#).expect("static regex");
    let mem_store =
        Regex::new(r#"^memory\s+store\s+"([^"]*)"\s*=\s*"([^"]*)"$"#).expect("static regex");
    let mem_find = Regex::new(r#"^memory\s+find\s+nearest\s+"([^"]*)"(?:\s+k=(\d+))?$"#)
        .expect("static regex");
    let mem_dump = Regex::new(r#"^memory\s+dump\s+(\S+)$"#).expect("static regex");

    let mut out = Vec::new();

    for (idx, raw) in source.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let kind = if let Some(rest) = line.strip_prefix("resolve ") {
            StmtKind::Resolve {
                name: rest.trim().to_string(),
            }
        } else if line.starts_with("describe ") {
            let caps = describe.captures(line).ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: format!("malformed describe: {}", line),
            })?;
            StmtKind::Describe {
                name: caps[1].to_string(),
                text: caps[2].to_string(),
            }
        } else if line.starts_with("spawn ") {
            let caps = spawn.captures(line).ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: format!("malformed spawn: {}", line),
            })?;
            StmtKind::Spawn {
                program: caps[1].to_string(),
                target: caps[2].to_string(),
            }
        } else if line == "navigate to penultimate" {
            StmtKind::NavigatePenultimate
        } else if line == "complete trajectory" {
            StmtKind::CompleteTrajectory
        } else if line.starts_with("memory create at") {
            let caps = s_coord.captures(line).ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: format!("expected S(k,t,e): {}", line),
            })?;
            let k: f64 = caps[1].parse().map_err(|_| ParseError::Syntax {
                line: line_no,
                message: format!("invalid number in coord: {}", &caps[1]),
            })?;
            let t: f64 = caps[2].parse().map_err(|_| ParseError::Syntax {
                line: line_no,
                message: format!("invalid number in coord: {}", &caps[2]),
            })?;
            let e: f64 = caps[3].parse().map_err(|_| ParseError::Syntax {
                line: line_no,
                message: format!("invalid number in coord: {}", &caps[3]),
            })?;
            let coord = SCoord::new(k, t, e).map_err(|err| ParseError::Syntax {
                line: line_no,
                message: err.to_string(),
            })?;
            StmtKind::MemoryCreate { coord }
        } else if line.starts_with("memory store") {
            let caps = mem_store.captures(line).ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: format!("malformed memory store: {}", line),
            })?;
            StmtKind::MemoryStore {
                name: caps[1].to_string(),
                text: caps[2].to_string(),
            }
        } else if line.starts_with("memory find nearest") {
            let caps = mem_find.captures(line).ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: format!("malformed memory find: {}", line),
            })?;
            let k = caps
                .get(2)
                .map(|m| m.as_str().parse::<usize>().unwrap_or(3))
                .unwrap_or(3);
            StmtKind::MemoryFind {
                query: caps[1].to_string(),
                k,
            }
        } else if line == "memory list" {
            StmtKind::MemoryList
        } else if line.starts_with("memory dump") {
            let caps = mem_dump.captures(line).ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: format!("malformed memory dump: {}", line),
            })?;
            StmtKind::MemoryDump {
                name: caps[1].to_string(),
            }
        } else if line == "demon sort" {
            StmtKind::DemonSort
        } else if line == "controller verify" {
            StmtKind::ControllerVerify
        } else if line == "kernel stats" {
            StmtKind::KernelStats
        } else if line == "kernel trace" {
            StmtKind::KernelTrace
        } else if line == "process list" {
            StmtKind::ProcessList
        } else {
            return Err(ParseError::Syntax {
                line: line_no,
                message: format!("unknown vaHera: {}", line),
            });
        };

        out.push(Stmt { kind });
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_describe_and_resolve() {
        let src = r#"
            describe ethanol with "ethanol C2H5OH bp 78"
            resolve ethanol
        "#;
        let stmts = parse_vahera(src).unwrap();
        assert_eq!(stmts.len(), 2);
        assert!(matches!(stmts[0].kind, StmtKind::Describe { .. }));
        assert!(matches!(stmts[1].kind, StmtKind::Resolve { .. }));
    }

    #[test]
    fn parse_memory_create_with_explicit_coord() {
        let stmts = parse_vahera("memory create at S(0.5, 0.4, 0.3)").unwrap();
        match &stmts[0].kind {
            StmtKind::MemoryCreate { coord } => {
                assert!((coord.k - 0.5).abs() < 1e-9);
                assert!((coord.t - 0.4).abs() < 1e-9);
                assert!((coord.e - 0.3).abs() < 1e-9);
            }
            _ => panic!("wrong kind"),
        }
    }

    #[test]
    fn parse_all_15_statement_types() {
        let src = r#"
describe a with "x"
resolve a
spawn p from a
navigate to penultimate
complete trajectory
memory create at S(0.1, 0.2, 0.3)
memory store "n" = "t"
memory find nearest "q" k=5
memory list
memory dump a
demon sort
controller verify
kernel stats
kernel trace
process list
"#;
        let stmts = parse_vahera(src).unwrap();
        assert_eq!(stmts.len(), 15);
    }

    #[test]
    fn comments_and_blanks_ignored() {
        let src = "# header\n\nresolve x\n# trailing\n";
        let stmts = parse_vahera(src).unwrap();
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn unknown_statement_is_error() {
        let err = parse_vahera("dance now").unwrap_err();
        assert!(matches!(err, ParseError::Syntax { .. }));
    }
}
