//! vaHera abstract syntax.

use buhera_substrate::SCoord;

/// One parsed vaHera statement.
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    /// What kind of statement it is.
    pub kind: StmtKind,
}

/// The 15 statement kinds supported in v0.1.0.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    /// `describe <name> with "<text>"`
    Describe { name: String, text: String },
    /// `resolve <name>`
    Resolve { name: String },
    /// `spawn <program> from <name>`
    Spawn { program: String, target: String },
    /// `navigate to penultimate`
    NavigatePenultimate,
    /// `complete trajectory`
    CompleteTrajectory,
    /// `memory create at S(k,t,e)`
    MemoryCreate { coord: SCoord },
    /// `memory store "<name>" = "<text>"`
    MemoryStore { name: String, text: String },
    /// `memory find nearest "<text>" k=<n>`
    MemoryFind { query: String, k: usize },
    /// `memory list`
    MemoryList,
    /// `memory dump <name>`
    MemoryDump { name: String },
    /// `demon sort`
    DemonSort,
    /// `controller verify`
    ControllerVerify,
    /// `kernel stats`
    KernelStats,
    /// `kernel trace`
    KernelTrace,
    /// `process list`
    ProcessList,

    // ── scientific-statement forms (sugar; each executes like an
    // existing kind above) ──
    /// `observed <name> as "<text>"` — executes like [`StmtKind::Describe`]
    Observed { name: String, text: String },
    /// `hypothesize <name>: "<text>"` — executes like [`StmtKind::Describe`]
    /// followed by [`StmtKind::Resolve`]
    Hypothesize { name: String, text: String },
    /// `run <program> on <name>` — executes like [`StmtKind::Spawn`]
    RunOn { program: String, target: String },
    /// `to completion` — executes like [`StmtKind::NavigatePenultimate`]
    /// followed by [`StmtKind::CompleteTrajectory`]
    ToCompletion,
    /// `compare <name> to "<text>" [k=<n>]` — executes like
    /// [`StmtKind::MemoryFind`]; `name` is carried for readability but not
    /// required by current kernel semantics
    CompareTo { name: String, query: String, k: usize },
    /// `record "<name>" = "<text>"` — executes like [`StmtKind::MemoryStore`]
    Record { name: String, text: String },
    /// `check consistency` — executes like [`StmtKind::ControllerVerify`]
    CheckConsistency,
    /// `rank by category` — executes like [`StmtKind::DemonSort`]
    RankByCategory,
}
