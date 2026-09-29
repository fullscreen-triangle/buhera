//! `buhera-pair` — the client half of gateway pairing, and the worker that
//! runs on the paired machine.
//!
//! The gateway (`buhera-gateway`) mints a catalyst token when a user clicks
//! "pair a machine" on the web (`long-grass` /pair page, `POST
//! /api/catalysts`). `buhera-pair pair --token <token> --name <name>` is what
//! the user runs *on the machine being paired* to hold that token, writing it
//! to a local config file and confirming it against the gateway.
//!
//! `buhera-pair run` is the other half: it dials the gateway's relay
//! (`GET /api/catalysts/relay`), holds the connection open, and executes
//! whatever vaHera work the gateway dispatches against a local
//! `buhera_kernel::Kernel` — see [`worker`]. The connection direction is
//! forced: this machine may sit behind NAT, so the gateway can never reach
//! it directly; it must be the one to dial out and hold the line open.

mod protocol;
mod worker;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

/// Default gateway. Overridable per-invocation with `--gateway`, so a
/// self-hosted or staging gateway can be paired against without a rebuild.
const DEFAULT_GATEWAY: &str = "https://buhera-91-98-157-147.sslip.io";

#[derive(Parser, Debug)]
#[command(
    name = "buhera-pair",
    version,
    about = "Pair this machine to a buhera-gateway account, and run its local worker."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Store a catalyst token minted on the web and confirm it works.
    Pair {
        /// The one-time token shown on the /pair page. Shown once there —
        /// paste it here right away.
        #[arg(long)]
        token: String,
        /// This machine's name, exactly as it was named when the token was
        /// minted on the /pair page — the relay handshake asserts it, since
        /// the token alone only proves the account, not which of its
        /// machines this is.
        #[arg(long)]
        name: String,
        /// Gateway base URL.
        #[arg(long, default_value = DEFAULT_GATEWAY)]
        gateway: String,
    },
    /// Show the currently stored pairing, if any.
    Status,
    /// Forget the stored pairing. Does not revoke the token on the gateway
    /// (there is no finer revocation yet than rotating the gateway's signing
    /// key, or unpairing the machine from the web /pair page) — this only
    /// clears the local copy.
    Forget,
    /// Hold the relay connection open and execute work dispatched to this
    /// machine, until interrupted (Ctrl-C).
    Run,
}

/// What's saved to disk after a successful pair.
#[derive(Debug, Serialize, Deserialize)]
struct StoredCredential {
    gateway: String,
    token: String,
    /// This machine's name. Absent in credentials saved by older `pair`
    /// invocations that took no `--name` — `run` refuses cleanly rather
    /// than guessing when it's missing, since the relay handshake needs it.
    #[serde(default)]
    name: Option<String>,
}

fn config_path() -> Result<PathBuf, String> {
    let dir = dirs::config_dir().ok_or("could not resolve a config directory for this OS")?;
    let dir = dir.join("buhera");
    std::fs::create_dir_all(&dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    Ok(dir.join("pair.json"))
}

fn load() -> Result<Option<StoredCredential>, String> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    serde_json::from_str(&raw)
        .map(Some)
        .map_err(|e| format!("parsing {}: {e}", path.display()))
}

fn save(cred: &StoredCredential) -> Result<(), String> {
    let path = config_path()?;
    let raw = serde_json::to_string_pretty(cred).expect("StoredCredential serializes");
    std::fs::write(&path, raw).map_err(|e| format!("writing {}: {e}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// `GET /api/catalysts/whoami`'s response — the machine-side counterpart of
/// a browser's session, confirming the catalyst token still works.
#[derive(Debug, Deserialize)]
struct WhoamiResponse {
    ok: bool,
    #[serde(default)]
    account_id: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

fn check_token(gateway: &str, token: &str) -> Result<String, String> {
    let client = reqwest::blocking::Client::new();
    let res = client
        .get(format!("{}/api/catalysts/whoami", gateway.trim_end_matches('/')))
        .bearer_auth(token)
        .send()
        .map_err(|e| format!("could not reach {gateway}: {e}"))?;

    let status = res.status();
    let body: WhoamiResponse = res
        .json()
        .map_err(|e| format!("gateway response was not the expected JSON: {e}"))?;

    if !status.is_success() || !body.ok {
        return Err(body.error.unwrap_or_else(|| format!("gateway rejected the token (HTTP {status})")));
    }
    body.account_id.ok_or_else(|| "gateway did not return an account id".to_string())
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Pair { token, name, gateway } => run_pair(&gateway, &token, &name),
        Command::Status => run_status(),
        Command::Forget => run_forget(),
        Command::Run => run_worker(),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run_pair(gateway: &str, token: &str, name: &str) -> Result<(), String> {
    println!("checking token against {gateway} …");
    let account_id = check_token(gateway, token)?;
    println!("token verified for account {account_id}");

    let cred = StoredCredential {
        gateway: gateway.to_string(),
        token: token.to_string(),
        name: Some(name.to_string()),
    };
    save(&cred)?;
    println!("paired as {name:?}. credential stored at {}", config_path()?.display());
    println!("run `buhera-pair run` to start accepting work on this machine.");
    Ok(())
}

fn run_status() -> Result<(), String> {
    match load()? {
        Some(cred) => {
            println!("paired to {}", cred.gateway);
            match check_token(&cred.gateway, &cred.token) {
                Ok(account_id) => println!("token is valid, account {account_id}."),
                Err(e) => println!("token check failed: {e}"),
            }
            match &cred.name {
                Some(name) => println!("machine name: {name}"),
                None => println!(
                    "no machine name on record (paired before --name existed) — \
                     re-run `buhera-pair pair` with --name to enable `run`."
                ),
            }
        }
        None => {
            println!("not paired. run `buhera-pair pair --token <token> --name <name>` with a token from the /pair page.")
        }
    }
    Ok(())
}

fn run_worker() -> Result<(), String> {
    let cred = load()?.ok_or("not paired. run `buhera-pair pair` first.")?;
    let name = cred
        .name
        .ok_or("this pairing has no machine name on record — re-run `buhera-pair pair` with --name.")?;

    let rt = tokio::runtime::Runtime::new().map_err(|e| format!("starting async runtime: {e}"))?;
    rt.block_on(async {
        tokio::select! {
            result = worker::run(&cred.gateway, &name, &cred.token) => result,
            _ = tokio::signal::ctrl_c() => {
                println!("\nstopping.");
                Ok(())
            }
        }
    })
}

fn run_forget() -> Result<(), String> {
    let path = config_path()?;
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| format!("removing {}: {e}", path.display()))?;
        println!("forgot the stored pairing.");
    } else {
        println!("nothing was paired.");
    }
    Ok(())
}
