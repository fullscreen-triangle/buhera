//! `buhera-pair` — the client half of gateway pairing.
//!
//! The gateway (`buhera-gateway`) mints a catalyst token when a user clicks
//! "pair a machine" on the web (`long-grass` /pair page, `POST
//! /api/catalysts`). This binary is what the user runs *on the machine being
//! paired* to hold that token: `buhera-pair pair --token <token> --name
//! <name>` writes it to a local config file and confirms it against the
//! gateway.
//!
//! This is deliberately small. The full relay — the gateway dialing back
//! into this machine to run work here — is not implemented yet (see
//! `buhera-gateway`'s `DEPLOYMENT.md`, "What is not here yet"). Until it
//! lands, this binary's job is just: hold the credential, prove it works.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

/// Default gateway. Overridable per-invocation with `--gateway`, so a
/// self-hosted or staging gateway can be paired against without a rebuild.
const DEFAULT_GATEWAY: &str = "https://buhera-91-98-157-147.sslip.io";

#[derive(Parser, Debug)]
#[command(name = "buhera-pair", about = "Pair this machine to a buhera-gateway account.")]
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
}

/// What's saved to disk after a successful pair.
#[derive(Debug, Serialize, Deserialize)]
struct StoredCredential {
    gateway: String,
    token: String,
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
        Command::Pair { token, gateway } => run_pair(&gateway, &token),
        Command::Status => run_status(),
        Command::Forget => run_forget(),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run_pair(gateway: &str, token: &str) -> Result<(), String> {
    println!("checking token against {gateway} …");
    let account_id = check_token(gateway, token)?;
    println!("token verified for account {account_id}");

    let cred = StoredCredential {
        gateway: gateway.to_string(),
        token: token.to_string(),
    };
    save(&cred)?;
    println!("paired. credential stored at {}", config_path()?.display());
    println!(
        "note: the gateway relay is not live yet — this machine holds the \
         credential, but the gateway cannot dispatch work to it until that \
         ships. see buhera-gateway's DEPLOYMENT.md."
    );
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
        }
        None => println!("not paired. run `buhera-pair pair --token <token>` with a token from the /pair page."),
    }
    Ok(())
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
