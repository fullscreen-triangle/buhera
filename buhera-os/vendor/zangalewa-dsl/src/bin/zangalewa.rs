//! `zangalewa` — the local CLI that lends your machine to the web tool.
//!
//! The deployed web tool has no model. Ollama runs here, on your machine, and
//! a browser cannot reach it: HTTPS pages may not fetch plaintext localhost,
//! there is no CORS grant, and a hosted server has no route to a private
//! host. This CLI closes that gap from the inside.
//!
//!   zangalewa connect                 # pair with the deployed tool
//!   zangalewa connect --url http://localhost:3000
//!   zangalewa generate vahera chunk "list all memories"   # no web tool at all
//!
//! ── Why polling rather than a server ──────────────────────────────────────
//!
//! This process makes only *outbound* requests. It opens no port, needs no
//! firewall rule, and is not reachable from the internet even while paired.
//! The web app can ask it for work only because it keeps asking the web app
//! whether there is any. That asymmetry is the entire security model, and it
//! is why this is a poll loop and not an HTTP server.
//!
//! ── What actually runs here ───────────────────────────────────────────────
//!
//! The whole generate loop — grounding, generation, validation against the
//! real compiler, and repair. The web app sends `(dslId, instructions)` and
//! receives a finished bag. It never sees a prompt and never judges a draft.
//! That is what makes this the Rust implementation rather than a proxy: the
//! TypeScript generator is not involved when a session is paired.

use std::time::Duration;
use zangalewa_dsl::{
    generate, list_dsls, ollama_reachable, provider_status, GenerateRequest,
};

const DEFAULT_URL: &str = "https://zangalewa-zoom-climb.vercel.app";

/// Per-job budget when the web tool does not set one. See the note at its use
/// site: sized for CPU-only inference, not for a GPU.
const LOCAL_TIMEOUT_MS: u64 = 600_000;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");

    let code = match cmd {
        "connect" => connect(&args).await,
        "generate" => generate_once(&args).await,
        "providers" => {
            print_providers().await;
            0
        }
        "help" | "--help" | "-h" => {
            print_help();
            0
        }
        other => {
            eprintln!("unknown command: {other}\n");
            print_help();
            2
        }
    };
    std::process::exit(code);
}

fn print_help() {
    println!(
        r#"zangalewa — writes DSL code, runs nothing.

USAGE
  zangalewa connect [--url URL]        Pair this machine with the web tool.
  zangalewa generate DSL EXTENT "..."  Generate once, print JSON, exit.
  zangalewa providers                  Show which models are reachable here.

CONNECT
  Prints a pair code. Type it into the web tool. This process then runs
  every generation the browser asks for, using your local models, and keeps
  running until you press Ctrl-C.

  It opens no port and accepts no inbound connection — it only ever calls
  out to the web app.

  --url   Override the web app. Default: {DEFAULT_URL}
          Use http://localhost:3000 against a dev server.

ENVIRONMENT
  OLLAMA_URL     default http://localhost:11434
  OLLAMA_MODEL   default llama3.2
  Cloud keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY) are used
  when set. Ollama is preferred when available because it is free."#
    );
}

/// Which providers this machine can genuinely use.
///
/// `provider_status()` is sync, so it reports Ollama as available whenever an
/// address exists — which is always. Here we can afford the round trip that
/// settles it, and the answer matters: it is what the CLI offers the web tool.
async fn reachable_providers() -> Vec<String> {
    let ollama_up = ollama_reachable().await;
    provider_status()
        .into_iter()
        .filter(|(id, _, available, _)| {
            *available && (*id != "ollama" || ollama_up)
        })
        .map(|(id, _, _, _)| id.to_string())
        .collect()
}

async fn print_providers() {
    let live = reachable_providers().await;
    println!("providers reachable from this machine:");
    for (id, label, _, cost) in provider_status() {
        let available = live.iter().any(|l| l == id);
        println!(
            "  {:<10} {:<10} {}  (cost {})",
            id,
            label,
            if available { "available" } else { "-" },
            cost
        );
    }
    println!("\ndsls this build can write:");
    for d in list_dsls() {
        println!(
            "  {:<12} fragments: {}",
            d.id,
            if d.accepts_fragment { "yes" } else { "no" }
        );
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).map(String::as_str)
}

// ── connect ───────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct SessionResponse {
    #[serde(rename = "sessionId")]
    session_id: String,
    #[serde(rename = "pairCode")]
    pair_code: String,
    #[serde(rename = "agentToken")]
    agent_token: String,
}

#[derive(serde::Deserialize)]
struct PollResponse {
    job: Option<PollJob>,
}

#[derive(serde::Deserialize)]
struct PollJob {
    id: String,
    payload: serde_json::Value,
}

async fn connect(args: &[String]) -> i32 {
    let base = flag(args, "--url").unwrap_or(DEFAULT_URL).trim_end_matches('/').to_string();

    // No timeout on the client itself: a poll legitimately hangs for ~25s and
    // a cold generation for ~2 minutes. Per-request timeouts are set where
    // they belong instead.
    let http = match reqwest::Client::builder().build() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("could not start http client: {e}");
            return 1;
        }
    };

    let dsls: Vec<String> = list_dsls().iter().map(|d| d.id.to_string()).collect();
    let providers = reachable_providers().await;

    if providers.is_empty() {
        eprintln!(
            "no model is reachable from this machine.\n\
             Start Ollama (`ollama serve`) or set an API key, then try again.\n\
             Run `zangalewa providers` to see what is detected."
        );
        return 1;
    }

    let hostname = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".into());

    let session: SessionResponse = match http
        .post(format!("{base}/api/bridge/session"))
        .json(&serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "hostname": hostname,
            "dsls": dsls,
            "providers": providers,
        }))
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .and_then(|r| r.error_for_status())
    {
        Ok(r) => match r.json().await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("the web app returned something unexpected: {e}");
                return 1;
            }
        },
        Err(e) => {
            eprintln!("could not reach {base}: {e}");
            eprintln!("If you meant a local dev server, pass --url http://localhost:3000");
            return 1;
        }
    };

    println!();
    println!("  pair code:  {}", session.pair_code);
    println!();
    println!("  Type it into {base}/zangalewa");
    println!("  Valid for 10 minutes, single use.");
    println!();
    println!("  models here: {}", providers.join(", "));
    println!("  Waiting for work. Ctrl-C to stop.");
    println!();

    poll_loop(&http, &base, &session.session_id, &session.agent_token).await
}

/// Poll, run, submit. Forever, until interrupted.
async fn poll_loop(http: &reqwest::Client, base: &str, session_id: &str, token: &str) -> i32 {
    // Transient network failures must not kill a long-running session, but a
    // genuinely dead endpoint should not be retried silently forever either.
    let mut consecutive_errors = 0u32;
    const MAX_CONSECUTIVE_ERRORS: u32 = 20;
    const SUBMIT_ATTEMPTS: u32 = 3;

    loop {
        let poll = http
            .post(format!("{base}/api/bridge/poll"))
            .bearer_auth(token)
            .json(&serde_json::json!({ "sessionId": session_id }))
            // Above the server's 25s hold, so the server ends the poll, not us.
            .timeout(Duration::from_secs(40))
            .send()
            .await;

        let response = match poll {
            Ok(r) if r.status() == reqwest::StatusCode::UNAUTHORIZED => {
                eprintln!("\nsession expired or was revoked — run `zangalewa connect` again");
                return 1;
            }
            Ok(r) => r,
            Err(e) => {
                consecutive_errors += 1;
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                    eprintln!("\ngiving up after {consecutive_errors} failed polls: {e}");
                    return 1;
                }
                // Back off gently; a laptop that slept should reconnect, not spin.
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        };
        consecutive_errors = 0;

        let body: PollResponse = match response.json().await {
            Ok(b) => b,
            Err(_) => continue, // malformed poll answer: just poll again
        };

        let Some(job) = body.job else { continue };

        // Run it. A panic here would drop the session, so failures are
        // reported back as job errors instead.
        let (result, error) = run_job(&job.payload).await;

        // Retried, because dropping it is the worst outcome available here:
        // the work is already done and paid for, and the browser is waiting
        // on a result that will now never arrive. A single transient failure
        // — a cold serverless instance, a route compiling for the first time
        // in dev — must not cost the whole generation.
        let body = serde_json::json!({
            "sessionId": session_id,
            "jobId": job.id,
            "result": result,
            "error": error,
        });

        let mut submitted = false;
        for attempt in 1..=SUBMIT_ATTEMPTS {
            match http
                .post(format!("{base}/api/bridge/result"))
                .bearer_auth(token)
                .json(&body)
                .timeout(Duration::from_secs(60))
                .send()
                .await
            {
                Ok(r) if r.status().is_success() => {
                    submitted = true;
                    break;
                }
                Ok(r) => {
                    // A 404 means the job or session is gone; retrying cannot
                    // bring it back.
                    eprintln!("result for job {} rejected: HTTP {}", job.id, r.status());
                    break;
                }
                Err(e) if attempt == SUBMIT_ATTEMPTS => {
                    eprintln!("could not submit result for job {}: {e}", job.id);
                }
                Err(_) => tokio::time::sleep(Duration::from_secs(2)).await,
            }
        }

        if !submitted {
            eprintln!("     the browser will time out waiting for that one");
        }
    }
}

/// Interpret a job payload and run it through the real generate loop.
async fn run_job(payload: &serde_json::Value) -> (Option<serde_json::Value>, Option<String>) {
    let dsl_id = payload.get("dslId").and_then(|v| v.as_str()).unwrap_or("");
    let instructions = payload
        .get("instructions")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if dsl_id.is_empty() || instructions.is_empty() {
        return (None, Some("job payload needs dslId and instructions".into()));
    }

    let extent = match payload.get("extent").and_then(|v| v.as_str()) {
        Some("chunk") => Some(zangalewa_dsl::Extent::Chunk),
        Some("script") => Some(zangalewa_dsl::Extent::Script),
        _ => None,
    };

    let req = GenerateRequest {
        dsl_id: dsl_id.to_string(),
        instructions: instructions.to_string(),
        extent,
        drafts: payload.get("drafts").and_then(|v| v.as_u64()).map(|n| n as u32),
        max_repairs: payload
            .get("maxRepairs")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32),
        model: payload
            .get("model")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        // The library default (200s) suits a GPU-backed model. This binary
        // runs wherever the user is, and on a CPU-only machine prompt
        // evaluation alone is the dominant cost: measured on an Intel UHD 620
        // laptop, where Ollama drops the integrated GPU and falls back to CPU,
        // the 1227-token grounding prompt processes at ~5 tok/s — about four
        // minutes before the first output token exists. 200s expired mid-
        // prompt and reported "no draft compiled", which reads as a model
        // failure when nothing had yet been asked of the model.
        //
        // Waiting is the right trade here: the alternative is not a faster
        // answer, it is no answer.
        timeout_ms: Some(
            payload
                .get("timeoutMs")
                .and_then(|v| v.as_u64())
                .unwrap_or(LOCAL_TIMEOUT_MS),
        ),
    };

    let label = format!("{} / {}", dsl_id, extent_label(extent));
    println!("  -> {label}: {}", truncate(instructions, 60));

    let started = std::time::Instant::now();
    let result = generate(req).await;
    let ms = started.elapsed().as_millis();

    if result.ok {
        println!("     {} chunk(s) in {ms}ms", result.chunks.len());
    } else {
        println!(
            "     refused in {ms}ms: {}",
            result.error.clone().unwrap_or_default()
        );
    }

    match serde_json::to_value(&result) {
        Ok(v) => (Some(v), None),
        Err(e) => (None, Some(format!("could not serialise result: {e}"))),
    }
}

fn extent_label(e: Option<zangalewa_dsl::Extent>) -> &'static str {
    match e {
        Some(zangalewa_dsl::Extent::Chunk) => "chunk",
        _ => "script",
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    let head: String = s.chars().take(n).collect();
    format!("{head}...")
}

// ── generate (no web tool involved) ───────────────────────────────────────

async fn generate_once(args: &[String]) -> i32 {
    let dsl_id = args.get(1).cloned().unwrap_or_else(|| "vahera".into());
    let extent = match args.get(2).map(String::as_str) {
        Some("chunk") => zangalewa_dsl::Extent::Chunk,
        _ => zangalewa_dsl::Extent::Script,
    };
    let Some(instructions) = args.get(3).cloned() else {
        eprintln!(r#"usage: zangalewa generate DSL EXTENT "instructions""#);
        return 2;
    };

    let result = generate(GenerateRequest {
        dsl_id,
        instructions,
        extent: Some(extent),
        ..Default::default()
    })
    .await;

    println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
    if result.ok {
        0
    } else {
        1
    }
}
