//! The `.wt` analysis DSL — parser, AST, and assertion evaluator.
//!
//! A `.wt` script is a *directed* analysis: instead of running every phase and
//! reading a wall of numbers, the author states what to analyse and what must
//! hold, and the run produces a verdict.
//!
//! ```text
//! scope PaymentFlow:
//!     include "src/payments/**"
//!
//! analyse:
//!     static
//!     cycles through ["checkout", "ledger"] max_depth 5
//!
//! assert:
//!     regime >= Coherent
//!     r_est >= 0.75
//!     no holonomy_violations
//!
//! report:
//!     format json
//! ```
//!
//! # Why this crate has no analysis dependencies
//!
//! It depends on none of `wt-graph`, `wt-static`, `wt-index`, or `wt-report`,
//! and it must stay that way. The browser sandbox compiles this crate to WASM
//! to give authors immediate parse errors and assertion feedback while typing —
//! with no local toolchain and no network round-trip.
//!
//! Analysis itself does *not* run in the browser. The measurement half needs to
//! walk a real source tree and is far heavier than a page should carry, so it
//! lives in the `wt` binary on the user's machine. The browser edits and
//! validates; the CLI computes. [`MetricView`] is the seam between them: the
//! browser can evaluate assertions against any metric it is handed, whoever
//! produced it.

pub mod ast;
pub mod error;
pub mod eval;
pub mod parse;

pub use ast::{
    Analyse, Assertion, CmpOp, Cycles, Dynamic, Format, Purpose, Report, ScalarField, Scope,
    Script,
};
pub use error::{ParseError, ParseErrorKind};
pub use eval::{evaluate, AssertionResult, EvalReport, MetricView, Outcome, Verdict};
pub use parse::parse;
