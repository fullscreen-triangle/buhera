//! Buhera OS demo.
//!
//! Boots a kernel (optionally pre-loaded with a JSON compound database)
//! and runs a vaHera script. With no arguments it runs a plain-text
//! notes demo that shows the OS storing sentences and retrieving them
//! by topic — no domain knowledge required.

use std::path::PathBuf;
use std::process::ExitCode;

use buhera_kernel::Kernel;
use buhera_os::{boot_os, load_nist, print_result};
use buhera_vahera::{execute_vahera, MoleculeDatabase};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "buhera-demo",
    about = "Boot Buhera OS and run a vaHera program."
)]
struct Args {
    /// Path to a vaHera source file. Defaults to the built-in notes
    /// demo, which doesn't need any external data.
    #[arg(long)]
    script: Option<PathBuf>,

    /// Optional JSON compound database. Provide this only if your
    /// script wants pre-allocated objects (e.g. `examples/demo.bvh`).
    #[arg(long)]
    data: Option<PathBuf>,

    /// Ternary-address depth used by CMM.
    #[arg(long, default_value_t = 12)]
    depth: usize,
}

/// Friendly plain-text default — same one in `examples/notes.bvh`.
const BUILTIN_SCRIPT: &str = r#"
# Plain-text notes demo: store some everyday sentences, then search
# them by topic. The OS files each sentence at its categorical address
# and finds matches by closeness in S-coordinate space.

memory store "weekend"   = "I need to do laundry and clean the kitchen this weekend"
memory store "groceries" = "buy milk eggs bread and coffee from the supermarket"
memory store "meeting"   = "schedule the team meeting for Wednesday afternoon"
memory store "email"     = "reply to the customer email about the late shipment"
memory store "exercise"  = "go for a run on Saturday morning before it gets hot"
memory store "book"      = "finish reading the chapter on bounded oscillatory systems"
memory store "code"      = "refactor the database connection pool to use async"
memory store "travel"    = "book a flight to Munich for the conference next month"

memory list

memory find nearest "shopping list" k=3
memory find nearest "morning workout" k=3
memory find nearest "travel plans" k=3

demon sort
kernel stats
"#;

fn main() -> ExitCode {
    let args = Args::parse();

    // Load NIST data only if explicitly asked for.
    let (mut kernel, molecules) = match args.data.as_ref() {
        Some(path) => match load_nist(path) {
            Ok(db) => {
                let kernel = boot_os(&db, args.depth);
                println!("compounds loaded from {}: {}", path.display(), db.molecules.len());
                (kernel, db.molecules)
            }
            Err(err) => {
                eprintln!("could not load {}: {}", path.display(), err);
                return ExitCode::from(2);
            }
        },
        None => (Kernel::new(args.depth), MoleculeDatabase::new()),
    };

    println!("buhera-os demo");
    println!("==============");
    println!("kernel depth: {}", args.depth);
    println!();

    let source = match args.script.as_ref() {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("could not read {}: {}", path.display(), err);
                return ExitCode::from(2);
            }
        },
        None => BUILTIN_SCRIPT.to_string(),
    };

    println!("vaHera source:");
    for line in source.lines() {
        if !line.trim().is_empty() {
            println!("  {}", line);
        }
    }
    println!();

    let ctx = match execute_vahera(&source, &mut kernel, &molecules) {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("vaHera execution failed: {}", err);
            return ExitCode::from(1);
        }
    };

    println!("interpreter trace:");
    for line in &ctx.trace {
        println!("  {}", line);
    }
    println!();

    println!("results:");
    for r in &ctx.results {
        print_result(r);
    }
    println!();

    let final_stats = kernel.stats();
    println!(
        "final: cmm_objects={} verified={} alerts={}",
        final_stats.cmm_objects, final_stats.pve.verified, final_stats.tem.alerts
    );

    ExitCode::SUCCESS
}
