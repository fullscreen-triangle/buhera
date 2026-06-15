//! Buhera OS NIST demo.
//!
//! Boots a kernel pre-loaded with NIST compounds, then runs either a
//! provided vaHera source file or a built-in script that asks property
//! questions through `describe → spawn → navigate → complete →
//! memory find nearest`.

use std::path::PathBuf;
use std::process::ExitCode;

use buhera_os::{boot_os, load_nist, print_result};
use buhera_vahera::execute_vahera;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "buhera-demo",
    about = "Boot the Buhera OS with NIST compounds and run a vaHera program."
)]
struct Args {
    /// Path to a vaHera source file to execute. Defaults to a built-in
    /// script demonstrating the standard NIST queries.
    #[arg(long)]
    script: Option<PathBuf>,

    /// Path to the NIST compound database (JSON).
    #[arg(long, default_value = "data/nist_compounds.json")]
    data: PathBuf,

    /// Ternary-address depth used by CMM.
    #[arg(long, default_value_t = 12)]
    depth: usize,
}

const BUILTIN_SCRIPT: &str = r#"
# Built-in demo: ask three questions about NIST compounds, then list
# everything and show the kernel's stats.
describe ethanol with "ethanol C2H5OH bp 78"
spawn lookup_eth from ethanol
navigate to penultimate
complete trajectory

memory find nearest "boiling point of benzene" k=3

memory list
kernel stats
"#;

fn main() -> ExitCode {
    let args = Args::parse();

    let db = match load_nist(&args.data) {
        Ok(db) => db,
        Err(err) => {
            eprintln!("could not load {}: {}", args.data.display(), err);
            return ExitCode::from(2);
        }
    };

    let mut kernel = boot_os(&db, args.depth);

    println!("buhera-os demo");
    println!("==============");
    println!("compounds loaded: {}", db.molecules.len());
    println!("kernel depth:     {}", args.depth);
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

    let ctx = match execute_vahera(&source, &mut kernel, &db.molecules) {
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
