//! Gateway entry point.
//!
//! Binds loopback by default. The gateway is meant to sit behind a TLS
//! reverse proxy, and binding a public interface would put an authentication
//! boundary directly on the internet on a machine whose provider applies no
//! network filtering — every bound port is immediately world-reachable. The
//! default is therefore the safe one, and exposing it is an explicit act.

use std::path::PathBuf;
use std::sync::Arc;

use buhera_gateway::http::{app, AppState, MIN_PASSWORD_LEN};
use buhera_gateway::store::{Store, StoreError};
use buhera_gateway::token::Signer;
use clap::Parser;
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(
    name = "buhera-gateway",
    about = "Buhera gateway: accounts, sessions, compute routing."
)]
struct Args {
    /// Address to bind. Keep on loopback behind a TLS proxy.
    #[arg(long, default_value = "127.0.0.1:8090")]
    bind: String,

    /// Path to the SQLite database holding accounts and catalysts.
    #[arg(long, default_value = "/var/lib/buhera/gateway.db")]
    db: PathBuf,

    /// Seed accounts from a JSON file and exit without starting the server.
    ///
    /// There is no public signup route (see `http` module docs) — this is
    /// the only way an account comes into existence. The file is a JSON
    /// array of `{"email": "...", "password": "..."}`. Accounts that
    /// already exist (by email) are left untouched, so the same file can
    /// be re-run safely. Never checked into the repo — it holds plaintext
    /// passwords until the moment they're hashed.
    #[arg(long)]
    seed: Option<PathBuf>,
}

/// One entry in a `--seed` file.
#[derive(Debug, Deserialize)]
struct SeedAccount {
    email: String,
    password: String,
}

/// Create every account in `path` that does not already exist.
///
/// Runs against the same `Store` the server itself would open, so it's
/// safe to point at the live database with the service stopped or running
/// (SQLite serialises the writes either way).
fn run_seed(db: &PathBuf, path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let accounts: Vec<SeedAccount> = serde_json::from_str(&raw)?;

    if let Some(dir) = db.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let store = Store::open(db)?;
    let now = buhera_gateway::now_unix();

    for a in &accounts {
        if a.password.len() < MIN_PASSWORD_LEN {
            eprintln!(
                "skipping {}: password shorter than {MIN_PASSWORD_LEN} characters",
                a.email
            );
            continue;
        }
        match store.create_account(&a.email, &a.password, now) {
            Ok(account) => println!("seeded {}", account.email),
            Err(StoreError::Duplicate) => println!("already exists, left alone: {}", a.email),
            Err(e) => eprintln!("failed to seed {}: {e}", a.email),
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "buhera_gateway=info,tower_http=warn".into()),
        )
        .init();

    let args = Args::parse();

    if let Some(seed_path) = &args.seed {
        run_seed(&args.db, seed_path)?;
        return Ok(());
    }

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

    if let Some(dir) = args.db.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let store = Store::open(&args.db)?;
    let state = Arc::new(AppState::new(store, signer));

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    let addr = listener.local_addr()?;
    if !addr.ip().is_loopback() {
        // Not fatal — there are legitimate reasons — but it should never
        // happen silently on a host with no network filtering.
        tracing::warn!(
            %addr,
            "bound a non-loopback address; ensure a firewall and TLS proxy are in front"
        );
    }
    tracing::info!(%addr, db = %args.db.display(), "buhera-gateway listening");

    axum::serve(listener, app(state))
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

/// Resolve on SIGTERM (systemd stop) or Ctrl-C.
async fn shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(e) => tracing::error!(error = %e, "cannot listen for SIGTERM"),
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("shutting down (ctrl-c)"),
        _ = term  => tracing::info!("shutting down (SIGTERM)"),
    }
}
