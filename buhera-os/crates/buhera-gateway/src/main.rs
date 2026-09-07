//! Gateway entry point.
//!
//! Binds loopback by default. The gateway is meant to sit behind a TLS
//! reverse proxy, and binding a public interface would put an authentication
//! boundary directly on the internet on a machine whose provider applies no
//! network filtering — every bound port is immediately world-reachable. The
//! default is therefore the safe one, and exposing it is an explicit act.

use std::path::PathBuf;
use std::sync::Arc;

use buhera_gateway::http::{app, AppState};
use buhera_gateway::store::Store;
use buhera_gateway::token::Signer;
use clap::Parser;

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
