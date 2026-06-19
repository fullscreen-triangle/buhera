//! Buhera OS demo.
//!
//! Boots a kernel and runs a vaHera script. By default uses the
//! semantic embedder (downloads a ~23 MB model on first run); pass
//! `--lexical` to use the dependency-free fallback.

use std::path::PathBuf;
use std::process::ExitCode;

use buhera_embed::{LexicalEmbedder, SemanticEmbedder, TextEmbedder};
use buhera_kernel::Kernel;
use buhera_os::{boot_os, load_nist, print_result, rerank_hits_with_overlap, EmbedderAdapter};
use buhera_vahera::{execute_vahera_with, MoleculeDatabase, NamedResult};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "buhera-demo",
    about = "Boot Buhera OS and run a vaHera program."
)]
struct Args {
    /// Path to a vaHera source file. Defaults to the built-in notes demo.
    #[arg(long)]
    script: Option<PathBuf>,

    /// Optional JSON compound database for `embed_molecule` lookups.
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

const BUILTIN_SCRIPT: &str = r#"
# Plain-text notes demo: store some everyday sentences, then search
# them by topic.

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

kernel stats
"#;

fn main() -> ExitCode {
    let args = Args::parse();

    // Pick an embedder.
    let embedder: Box<dyn TextEmbedder> = if args.lexical {
        eprintln!("(using lexical embedder)");
        Box::new(LexicalEmbedder::new())
    } else {
        eprintln!("(loading semantic embedder; first run downloads ~133 MB)");
        match SemanticEmbedder::new() {
            Ok(e) => {
                eprintln!("(semantic embedder ready: {})", e.name());
                Box::new(e)
            }
            Err(err) => {
                eprintln!("(semantic embedder failed: {}; falling back to lexical)", err);
                Box::new(LexicalEmbedder::new())
            }
        }
    };

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

    let adapter = EmbedderAdapter::new(embedder.as_ref());
    let mut ctx = match execute_vahera_with(&source, &mut kernel, &molecules, &adapter) {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("vaHera execution failed: {}", err);
            return ExitCode::from(1);
        }
    };

    // Apply token-overlap re-ranking to every FindHits result, using
    // each hit's own originating query.
    if !args.no_overlap {
        for r in &mut ctx.results {
            if let NamedResult::FindHits { query, hits } = r {
                rerank_hits_with_overlap(query, hits, 0.5);
            }
        }
    }

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
