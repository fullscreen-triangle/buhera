//! Command-line driver for the Purpose MVP.
//!
//! Usage:
//!     purpose query "Tell me about SOD1"
//!     purpose query "Tell me about SOD1" --raw         # JSON Value only
//!     purpose query "Tell me about SOD1" --fragment    # compiled vaHera, no execution
//!     purpose operations                                # list registered ops
//!
//! The `--raw` mode is the wire contract consumed by the web tool and by
//! any host that wraps Purpose as a subprocess (integration.md §5, §9.4).

use std::process::ExitCode;

use clap::{Parser, Subcommand};
use purpose_core::{Resolver, VaHera};
use purpose_domains_protein as protein;
use purpose_operations::{Executor, OperationRegistry};

#[derive(Parser, Debug)]
#[command(name = "purpose", version, about = "Purpose compilation + execution driver")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compile an utterance to a vaHera fragment and execute it.
    Query {
        /// The natural-language utterance to compile.
        utterance: String,
        /// Emit only the executed `Value` as JSON to stdout (no decorations).
        #[arg(long)]
        raw: bool,
        /// Stop after compilation; emit the vaHera fragment instead of executing.
        #[arg(long)]
        fragment: bool,
    },
    /// List every registered operation's signature.
    Operations,
    /// Emit a self-describing JSON record of the loaded domains.
    Introspect,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::from(1)
        }
    }
}

async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let mut registry = OperationRegistry::new();
    protein::register_providers(&mut registry);
    let executor = Executor::new(registry);

    let domain = protein::domain();

    match cli.command {
        Command::Query {
            utterance,
            raw,
            fragment,
        } => {
            let frag: VaHera = domain.resolver.compile(&utterance).await?;

            if fragment {
                let json = serde_json::to_string_pretty(&frag)?;
                println!("{}", json);
                return Ok(());
            }

            let value = executor.execute(&frag).await?;
            let json = serde_json::to_string(&value)?;
            if raw {
                println!("{}", json);
            } else {
                println!("utterance: {}", utterance);
                println!("fragment:  {}", serde_json::to_string(&frag)?);
                println!("value:     {}", json);
            }
        }
        Command::Operations => {
            for op in executor.registry().operations() {
                println!("{}  {}", op.name, op.description);
            }
        }
        Command::Introspect => {
            let record = serde_json::json!({
                "domain": domain.name,
                "operations": executor
                    .registry()
                    .operations()
                    .map(|op| serde_json::json!({
                        "name": op.name,
                        "description": op.description,
                        "output": format!("{:?}", op.output),
                    }))
                    .collect::<Vec<_>>(),
            });
            println!("{}", serde_json::to_string_pretty(&record)?);
        }
    }

    Ok(())
}
