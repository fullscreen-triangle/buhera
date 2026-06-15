//! Buhera OS interactive vaHera REPL.
//!
//! Reads one vaHera statement per line from stdin, dispatches it against
//! a live kernel, prints results, and loops. Type `:help` for a list of
//! statement kinds, `:quit` or Ctrl-D to exit.

use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use buhera_kernel::Kernel;
use buhera_os::{boot_os, load_nist, print_result, NistDatabase};
use buhera_vahera::{execute_vahera, MoleculeDatabase};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "buhera-repl",
    about = "Interactive vaHera REPL against a live Buhera kernel."
)]
struct Args {
    /// Optional path to a NIST compound database. If absent, the REPL
    /// starts with an empty kernel.
    #[arg(long)]
    data: Option<PathBuf>,

    /// Ternary-address depth used by CMM.
    #[arg(long, default_value_t = 12)]
    depth: usize,
}

const HELP: &str = r#"
vaHera statements:
  describe <name> with "<text>"
  resolve <name>
  spawn <program> from <name>
  navigate to penultimate
  complete trajectory
  memory create at S(<k>,<t>,<e>)
  memory store "<name>" = "<text>"
  memory find nearest "<text>" k=<n>
  memory list
  memory dump <name>
  demon sort
  controller verify
  kernel stats
  kernel trace
  process list

REPL commands:
  :help    show this message
  :quit    exit the REPL
"#;

fn main() -> ExitCode {
    let args = Args::parse();

    let db = match args.data.as_ref() {
        Some(p) => match load_nist(p) {
            Ok(db) => Some(db),
            Err(err) => {
                eprintln!("could not load {}: {}", p.display(), err);
                return ExitCode::from(2);
            }
        },
        None => None,
    };

    let molecules: MoleculeDatabase = db
        .as_ref()
        .map(|d| d.molecules.clone())
        .unwrap_or_default();
    let mut kernel = match &db {
        Some(d) => boot_os(d, args.depth),
        None => Kernel::new(args.depth),
    };

    println!("buhera-os repl  (depth={}, compounds={})", args.depth, kernel.cmm.len());
    println!("type :help for commands, :quit to exit");

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut input = stdin.lock();

    loop {
        write!(stdout, "vahera> ").ok();
        stdout.flush().ok();

        let mut line = String::new();
        match input.read_line(&mut line) {
            Ok(0) => {
                println!();
                return ExitCode::SUCCESS;
            }
            Ok(_) => {}
            Err(err) => {
                eprintln!("read error: {}", err);
                return ExitCode::from(1);
            }
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == ":quit" || trimmed == ":exit" {
            return ExitCode::SUCCESS;
        }
        if trimmed == ":help" {
            print!("{}", HELP);
            continue;
        }

        match execute_vahera(trimmed, &mut kernel, &molecules) {
            Ok(ctx) => {
                for r in &ctx.results {
                    print_result(r);
                }
                if ctx.results.is_empty() {
                    // For statements that don't produce a NamedResult,
                    // print the interpreter's own trace line.
                    if let Some(t) = ctx.trace.last() {
                        println!("  {}", t);
                    } else {
                        println!("  ok");
                    }
                }
            }
            Err(err) => {
                eprintln!("error: {}", err);
            }
        }
    }
}

// Silence unused-import warning when the conditional `db` arm is `None`.
#[allow(dead_code)]
fn _unused(_d: &NistDatabase, _m: &BTreeMap<String, ()>) {}
