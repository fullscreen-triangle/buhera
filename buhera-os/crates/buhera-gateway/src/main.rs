//! Gateway entry point.
//!
//! Binds loopback by default. The gateway is meant to sit behind a TLS
//! reverse proxy (Caddy on this host), and binding `0.0.0.0` would put an
//! authentication boundary directly on the public internet on a machine
//! whose provider applies no network filtering — every bound port is
//! immediately world-reachable. The default is therefore the safe one and
//! exposing it is an explicit act.

use std::path::PathBuf;

use buhera_gateway::token::Signer;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "buhera-gateway", about = "Buhera gateway: accounts, sessions, compute routing.")]
struct Args {
    /// Address to bind. Keep on loopback behind a TLS proxy.
    #[arg(long, default_value = "127.0.0.1:8090")]
    bind: String,

    /// Path to the SQLite database holding accounts and catalysts.
    #[arg(long, default_value = "/var/lib/buhera/gateway.db")]
    db: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // The signing key comes from the environment, never from a flag:
    // a flag lands in the process table and in shell history.
    let signer = match std::env::var("BUHERA_GATEWAY_KEY") {
        Ok(k) => Signer::from_base64(&k)?,
        Err(_) => {
            eprintln!(
                "BUHERA_GATEWAY_KEY is not set.\n\
                 Generate one and store it where only this service can read it:\n\n    \
                 {}\n",
                Signer::generate().to_base64()
            );
            std::process::exit(2);
        }
    };

    let store = buhera_gateway::store::Store::open(&args.db)?;

    // Deliberately not yet serving: the HTTP surface lands next, on top of
    // these three verified pieces. Report what is wired so a deploy of this
    // revision is honest about what it does.
    println!("buhera-gateway");
    println!("  bind (pending http): {}", args.bind);
    println!("  database:            {}", args.db.display());
    println!("  signing key:         loaded ({:?})", signer);
    let _ = store;
    Ok(())
}
