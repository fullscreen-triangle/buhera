//! Parse errors, carrying the line that produced them.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParseError {
    /// 1-based line number. `0` means the error is about the script as a whole
    /// (e.g. a missing `scope` block) and belongs to no single line.
    pub line: usize,
    pub kind: ParseErrorKind,
}

impl ParseError {
    pub fn new(line: usize, kind: ParseErrorKind) -> Self {
        ParseError { line, kind }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 {
            write!(f, "{}", self.kind)
        } else {
            write!(f, "line {}: {}", self.line, self.kind)
        }
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParseErrorKind {
    UnknownBlock(String),
    DirectiveOutsideBlock,
    MissingScope,
    UnknownDirective { block: String, found: String },
    UnknownAssertion(String),
    UnknownRegime(String),
    UnknownFormat(String),
    MissingArgument(String),
    BadNumber(String),
    BadOperator(String),
    ExpectedList(String),
    UnclosedList,
    UnexpectedTrailing(String),
}

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ParseErrorKind::*;
        match self {
            UnknownBlock(s) =>
                write!(f, "unknown block `{s}` (expected scope, analyse, assert, or report)"),
            DirectiveOutsideBlock =>
                write!(f, "indented directive before any block header"),
            MissingScope =>
                write!(f, "script has no `scope` block"),
            UnknownDirective { block, found } =>
                write!(f, "`{found}` is not a directive of the `{block}` block"),
            UnknownAssertion(s) =>
                write!(f, "unknown assertion `{s}`"),
            UnknownRegime(s) =>
                write!(f, "unknown regime `{s}` (expected Turbulent, ApertureDominated, \
                           HierarchicalCascade, Coherent, or PhaseLocked)"),
            UnknownFormat(s) =>
                write!(f, "unknown report format `{s}` (expected text or json)"),
            MissingArgument(what) =>
                write!(f, "missing {what}"),
            BadNumber(s) =>
                write!(f, "`{s}` is not a number"),
            BadOperator(s) =>
                write!(f, "`{s}` is not a comparison operator (expected >=, >, <=, <, or ==)"),
            ExpectedList(s) =>
                write!(f, "expected a `[...]` list, found `{s}`"),
            UnclosedList =>
                write!(f, "list is missing its closing `]`"),
            UnexpectedTrailing(s) =>
                write!(f, "unexpected trailing input `{s}`"),
        }
    }
}
