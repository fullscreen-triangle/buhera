//! vaHera abstract syntax.

use buhera_substrate::SCoord;

/// One parsed vaHera statement.
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    /// What kind of statement it is.
    pub kind: StmtKind,
}

/// The 15 statement kinds supported in v0.1.0.
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
}
