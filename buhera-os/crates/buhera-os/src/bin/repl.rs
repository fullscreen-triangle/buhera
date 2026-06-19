//! Buhera OS interactive vaHera REPL.
//!
//! Reads one line of input at a time. Lines starting with a vaHera
//! keyword are parsed and dispatched; bare natural-language lines are
//! treated as a `memory find nearest` query. `:help` shows the menu,
//! `:tour` runs a guided demo, `:quit` exits.

use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use buhera_embed::{LexicalEmbedder, SemanticEmbedder, TextEmbedder};
use buhera_kernel::Kernel;
use buhera_os::{boot_os, load_nist, print_result, rerank_hits_with_overlap, EmbedderAdapter, NistDatabase};
use buhera_vahera::{execute_vahera_with, MoleculeDatabase, NamedResult};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "buhera-repl",
    about = "Interactive vaHera REPL against a live Buhera kernel."
)]
struct Args {
    /// Optional path to a JSON compound database. If absent, the REPL
    /// starts with an empty kernel — perfect for trying out the OS.
    #[arg(long)]
    data: Option<PathBuf>,

    /// Ternary-address depth used by CMM.
    #[arg(long, default_value_t = 12)]
    depth: usize,

    /// Use the dependency-free lexical embedder instead of the
    /// semantic model.
    #[arg(long, default_value_t = false)]
    lexical: bool,

    /// Disable the token-overlap re-ranking pass.
    #[arg(long, default_value_t = false)]
    no_overlap: bool,
}

const HELP: &str = r#"
The Buhera OS files things by their categorical address — three numbers
between 0 and 1 derived from the content of a piece of text. You store
notes; the OS files them by address. You ask a question; the OS finds
the notes whose addresses are closest to the question's address.

Quick-start (just type these one at a time):

  store apple = "I bought apples at the market"
  store work  = "the deadline is Friday afternoon"
  find "shopping"
  find "deadline"
  list
  stats

Full vaHera statements (the underlying language):

  describe <name> with "<text>"            bind a name to a coordinate
  resolve <name>                           recompute the coordinate
  spawn <program> from <name>              create a categorical process
  navigate to penultimate                  walk back to one step short
  complete trajectory                      apply the completion morphism
  memory create at S(<k>,<t>,<e>)          allocate at an explicit coord
  memory store "<name>" = "<text>"         store text at content coord
  memory find nearest "<text>" k=<n>       categorical retrieval
  memory list                              list everything
  memory dump <name>                       show one object in detail
  demon sort                               zero-cost categorical sort
  controller verify                        triple-equivalence diagnostics
  kernel stats                             per-subsystem statistics
  kernel trace                             activity log
  process list                             all spawned processes

REPL shortcuts (more typing-friendly):

  store <name> = "<text>"     same as: memory store "<name>" = "<text>"
  find "<text>"               same as: memory find nearest "<text>" k=3
  list                        same as: memory list
  dump <name>                 same as: memory dump <name>
  sort                        same as: demon sort
  stats                       same as: kernel stats
  trace                       same as: kernel trace
  procs                       same as: process list

A bare line with no keyword is treated as a search query.

Commands:

  :help    show this message
  :tour    run a short guided demo
  :clear   start over with an empty kernel
  :quit    exit
"#;

const TOUR: &str = r#"# A short tour of the Buhera OS.
memory store "lunch"   = "let's grab pizza for lunch today"
memory store "task"    = "remember to update the project README"
memory store "weather" = "it might rain this afternoon"
memory find nearest "what's for food" k=2
memory find nearest "writing" k=2
memory list
kernel stats
"#;

fn main() -> ExitCode {
    let args = Args::parse();

    // Pick an embedder.
    let embedder: Box<dyn TextEmbedder> = if args.lexical {
        Box::new(LexicalEmbedder::new())
    } else {
        eprintln!("(loading semantic embedder; first run downloads ~23 MB)");
        match SemanticEmbedder::new() {
            Ok(e) => Box::new(e),
            Err(err) => {
                eprintln!("(semantic embedder failed: {}; falling back to lexical)", err);
                Box::new(LexicalEmbedder::new())
            }
        }
    };

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

    println!(
        "buhera-os repl  (depth={}, embedder={}, objects loaded={})",
        args.depth,
        embedder.name(),
        kernel.cmm.len()
    );
    println!("type something to search, or :help for the menu, :quit to exit");

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut input = stdin.lock();

    loop {
        write!(stdout, "buhera> ").ok();
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

        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }

        // REPL meta-commands.
        match trimmed.as_str() {
            ":quit" | ":exit" => return ExitCode::SUCCESS,
            ":help" => {
                print!("{}", HELP);
                continue;
            }
            ":clear" => {
                kernel = Kernel::new(args.depth);
                println!("(kernel reset)");
                continue;
            }
            ":tour" => {
                let adapter = EmbedderAdapter::new(embedder.as_ref());
                match execute_vahera_with(TOUR, &mut kernel, &molecules, &adapter) {
                    Ok(mut ctx) => {
                        if !args.no_overlap {
                            for r in &mut ctx.results {
                                if let NamedResult::FindHits { query, hits } = r {
                                    rerank_hits_with_overlap(query, hits, 0.5);
                                }
                            }
                        }
                        for line in &ctx.trace {
                            println!("  {}", line);
                        }
                        for r in &ctx.results {
                            print_result(r);
                        }
                    }
                    Err(err) => eprintln!("error during tour: {}", err),
                }
            }
            _ => {
                // Normal input: try a friendly shortcut first, then fall
                // back to full vaHera.
                let vahera = translate_shortcut(&trimmed);
                let adapter = EmbedderAdapter::new(embedder.as_ref());
                match execute_vahera_with(&vahera, &mut kernel, &molecules, &adapter) {
                    Ok(mut ctx) => {
                        if !args.no_overlap {
                            for r in &mut ctx.results {
                                if let NamedResult::FindHits { query, hits } = r {
                                    rerank_hits_with_overlap(query, hits, 0.5);
                                }
                            }
                        }
                        for r in &ctx.results {
                            print_result(r);
                        }
                        if ctx.results.is_empty() {
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
    }
}

/// Translate a friendly REPL input into vaHera source.
///
/// Falls through unchanged for anything that already looks like a real
/// vaHera statement. Otherwise:
///
/// * `store <name> = "<text>"`            -> `memory store "<name>" = "<text>"`
/// * `find "<text>"` / `find "<text>" k=N` -> `memory find nearest "<text>" k=N`
/// * `list`                                -> `memory list`
/// * `dump <name>`                         -> `memory dump <name>`
/// * `sort` / `stats` / `trace` / `procs`  -> their `kernel`/`demon`/`process` equivalents
/// * Anything else (no leading keyword)    -> `memory find nearest "<input>" k=3`
fn translate_shortcut(input: &str) -> String {
    let lower = input.trim().to_lowercase();

    // Already a vaHera statement — pass through.
    let vahera_prefixes = [
        "describe ",
        "resolve ",
        "spawn ",
        "navigate ",
        "complete ",
        "memory ",
        "demon ",
        "controller ",
        "kernel ",
        "process ",
    ];
    if vahera_prefixes.iter().any(|p| lower.starts_with(p)) {
        return input.to_string();
    }

    // store <name> = "<text>"
    if strip_keyword(&lower, "store ").is_some() {
        let original_rest = &input.trim()[6..];
        if let Some(eq) = original_rest.find('=') {
            let (name_part, value_part) = original_rest.split_at(eq);
            let name = name_part.trim();
            let value = value_part[1..].trim();
            let value = strip_quotes(value);
            if !name.is_empty() && !value.is_empty() {
                return format!("memory store \"{}\" = \"{}\"", name, value);
            }
        }
    }

    // find "<text>" k=N    or    find "<text>"
    if strip_keyword(&lower, "find ").is_some() {
        let original_rest = &input.trim()[5..];
        let (text, k) = split_find_args(original_rest);
        let t = strip_quotes(text.trim());
        return format!("memory find nearest \"{}\" k={}", t, k);
    }

    // Single-word shortcuts.
    if let Some(rest) = strip_keyword(&lower, "dump ") {
        return format!("memory dump {}", rest.trim());
    }

    match lower.as_str() {
        "list" => return "memory list".to_string(),
        "sort" => return "demon sort".to_string(),
        "stats" => return "kernel stats".to_string(),
        "trace" => return "kernel trace".to_string(),
        "procs" | "ps" => return "process list".to_string(),
        "verify" => return "controller verify".to_string(),
        _ => {}
    }

    // Fall through: treat the whole line as a search query.
    format!(
        "memory find nearest \"{}\" k=3",
        input.trim().replace('"', "'")
    )
}

fn strip_keyword<'a>(s: &'a str, kw: &str) -> Option<&'a str> {
    s.strip_prefix(kw)
}

fn strip_quotes(s: &str) -> &str {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

fn split_find_args(input: &str) -> (&str, usize) {
    // Look for ` k=` suffix at the end.
    if let Some(k_pos) = input.rfind(" k=") {
        let k_str = &input[k_pos + 3..].trim();
        if let Ok(k) = k_str.parse::<usize>() {
            return (&input[..k_pos], k);
        }
    }
    (input, 3)
}

// Silence unused-import warning when no NIST data is provided.
#[allow(dead_code)]
fn _unused(_d: &NistDatabase, _m: &BTreeMap<String, ()>) {}
