//! vaHera — the Buhera OS's internal declarative language.
//!
//! v0.1.0 covers 15 statement types (full Python parity from
//! `driven/system/vahera/interpreter.py` plus five inspection commands):
//!
//! | Statement                                  | Effect                                   |
//! |--------------------------------------------|------------------------------------------|
//! | `describe <name> with "<text>"`            | Bind text content to a categorical coord |
//! | `resolve <name>`                           | Compute / look up the coord for `<name>` |
//! | `spawn <program> from <name>`              | Create a categorical process to `<name>` |
//! | `navigate to penultimate`                  | Backward-navigate to the penultimate     |
//! | `complete trajectory`                      | Apply the completion morphism            |
//! | `memory create at S(<k>,<t>,<e>)`          | Allocate at an explicit coord            |
//! | `memory store "<name>" = "<text>"`         | Store text at its content coord          |
//! | `memory find nearest "<text>" k=<n>`       | Categorical retrieval                    |
//! | `memory list`                              | List all allocated objects               |
//! | `memory dump <name>`                       | Print payload + coord + address of one   |
//! | `demon sort`                               | Zero-cost categorical sort               |
//! | `controller verify`                        | Triple-equivalence diagnostics           |
//! | `kernel stats`                             | Per-subsystem statistics                 |
//! | `kernel trace`                             | Activity log                             |
//! | `process list`                             | All spawned processes and their states   |

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// Field-level docs on the simple data-carrying StmtKind variants are
// not load-bearing; the variants themselves are documented.
#![allow(clippy::missing_docs_in_private_items)]

mod ast;
mod parser;
mod interpreter;

pub use ast::{Stmt, StmtKind};
pub use parser::{parse_vahera, ParseError};
pub use interpreter::{
    execute_vahera, ExecContext, ExecError, MoleculeDatabase, NamedResult, ResolvedTarget,
};
