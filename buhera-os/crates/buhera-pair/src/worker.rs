//! `buhera-pair run` — hold the relay connection open and execute work.
//!
//! Connects out to the gateway (the direction is forced: this machine sits
//! behind whatever NAT its home network has, so the gateway can never dial
//! it — see `buhera-gateway`'s module docs), announces itself, and then
//! answers every `Run` frame against one long-lived local `Kernel`, exactly
//! the pattern `buhera-gateway`'s own degraded-path `session::Sessions`
//! uses for its gateway-side kernel — same `execute_vahera`/`render_result`
//! functions, same ternary-address depth, so a statement means the same
//! thing here as it does there.

use std::time::Duration;

use buhera_kernel::Kernel;
use buhera_vahera::{execute_vahera, render_result as render, MoleculeDatabase};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

use crate::protocol::{ClientMsg, ServerMsg};

/// Ternary-address depth. Must match `buhera_gateway::session::DEPTH` and
/// `buhera_kernel::Kernel::with_default_depth`'s `12` — a different value
/// would silently change the addresses a query resolves to, breaking parity
/// between what the same statement means locally versus on the gateway's
/// own degraded path.
const DEPTH: usize = 12;

/// How often to send a heartbeat. Matches the interval
/// `buhera_gateway::router::LIVENESS_WINDOW_SECS`'s doc comment assumes
/// (60s window, tolerating three consecutive misses at this cadence).
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);

/// Reconnect backoff steps, capped. No jitter: at the scale of one client
/// reconnecting to one gateway, thundering-herd isn't a concern.
const RECONNECT_BACKOFF: [Duration; 4] =
    [Duration::from_secs(1), Duration::from_secs(2), Duration::from_secs(5), Duration::from_secs(10)];

/// Run forever: connect, serve, and on any disconnect, reconnect with
/// backoff. Returns only on a fatal, non-recoverable error (a malformed
/// gateway URL, for instance) — a dropped connection is not fatal, it is
/// the expected shape of "the network hiccuped" and is retried.
pub async fn run(gateway: &str, name: &str, token: &str) -> Result<(), String> {
    let ws_url = to_ws_url(gateway)?;
    let mut backoff = RECONNECT_BACKOFF.iter().cycle();

    // One kernel for the life of this process — reused across reconnects,
    // exactly as `session::Sessions` reuses one kernel per account across
    // requests. A dropped connection should not erase what was stored.
    let mut kernel = Kernel::new(DEPTH);
    let molecules = MoleculeDatabase::new();

    loop {
        println!("connecting to {ws_url} as {name:?} …");
        match connect_and_serve(&ws_url, name, token, &mut kernel, &molecules).await {
            Ok(()) => {
                // The gateway closed the socket cleanly (not expected in
                // normal operation, but not an error either) — reconnect.
                println!("connection closed; reconnecting …");
            }
            Err(e) => {
                eprintln!("relay connection error: {e}");
            }
        }
        let delay = *backoff.next().expect("cycle never ends");
        tokio::time::sleep(delay).await;
    }
}

fn to_ws_url(gateway: &str) -> Result<String, String> {
    let mut url = url::Url::parse(gateway).map_err(|e| format!("invalid gateway URL {gateway:?}: {e}"))?;
    let scheme = match url.scheme() {
        "https" => "wss",
        "http" => "ws",
        other => return Err(format!("gateway URL has an unsupported scheme {other:?}")),
    };
    url.set_scheme(scheme).map_err(|_| "could not rewrite the URL scheme".to_string())?;
    url.set_path("/api/catalysts/relay");
    Ok(url.to_string())
}

async fn connect_and_serve(
    ws_url: &str,
    name: &str,
    token: &str,
    kernel: &mut Kernel,
    molecules: &MoleculeDatabase,
) -> Result<(), String> {
    let mut request = ws_url.into_client_request().map_err(|e| format!("building request: {e}"))?;
    request
        .headers_mut()
        .insert("Authorization", format!("Bearer {token}").parse().map_err(|e| format!("{e}"))?);

    let (socket, _response) = tokio_tungstenite::connect_async(request)
        .await
        .map_err(|e| format!("connecting: {e}"))?;
    let (mut sink, mut stream) = socket.split();

    send(&mut sink, &ClientMsg::Hello { name: name.to_string() }).await?;

    // Confirm the gateway actually accepted the Hello before declaring
    // ourselves connected — an unrecognized machine name gets the socket
    // closed rather than a HelloAck, and that should be reported plainly
    // rather than silently sitting in a heartbeat loop that can never work.
    match stream.next().await {
        Some(Ok(Message::Text(text))) => match serde_json::from_str::<ServerMsg>(&text) {
            Ok(ServerMsg::HelloAck) => println!("connected."),
            Ok(other) => return Err(format!("expected HelloAck, got {other:?}")),
            Err(e) => return Err(format!("unparseable response to Hello: {e}")),
        },
        Some(Ok(_)) => return Err("expected a text frame acknowledging Hello".to_string()),
        Some(Err(e)) => return Err(format!("reading Hello response: {e}")),
        None => return Err("gateway closed the connection before acknowledging Hello".to_string()),
    }

    let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
    heartbeat.tick().await; // first tick fires immediately; skip it, Hello already announced liveness

    loop {
        tokio::select! {
            _ = heartbeat.tick() => {
                send(&mut sink, &ClientMsg::Heartbeat).await?;
            }
            frame = stream.next() => {
                match frame {
                    Some(Ok(Message::Text(text))) => {
                        let msg: ServerMsg = match serde_json::from_str(&text) {
                            Ok(m) => m,
                            Err(e) => {
                                eprintln!("unparseable relay frame from gateway: {e}");
                                continue;
                            }
                        };
                        match msg {
                            ServerMsg::HelloAck => {
                                // Only expected once, already consumed above.
                                continue;
                            }
                            ServerMsg::Run { request_id, source } => {
                                let reply = execute_locally(kernel, molecules, &source);
                                send(&mut sink, &reply_to(request_id, reply)).await?;
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => return Ok(()),
                    Some(Ok(_)) => continue,
                    Some(Err(e)) => return Err(format!("relay connection lost: {e}")),
                }
            }
        }
    }
}

fn execute_locally(
    kernel: &mut Kernel,
    molecules: &MoleculeDatabase,
    source: &str,
) -> Result<(Vec<serde_json::Value>, Vec<String>), String> {
    let ctx = execute_vahera(source, kernel, molecules).map_err(|e| e.to_string())?;
    Ok((ctx.results.iter().map(render).collect(), ctx.trace))
}

fn reply_to(request_id: String, outcome: Result<(Vec<serde_json::Value>, Vec<String>), String>) -> ClientMsg {
    match outcome {
        Ok((results, trace)) => ClientMsg::RunResult { request_id, results, trace },
        Err(message) => ClientMsg::RunError { request_id, message },
    }
}

async fn send<S>(sink: &mut S, msg: &ClientMsg) -> Result<(), String>
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::fmt::Display,
{
    let text = serde_json::to_string(msg).expect("ClientMsg serializes");
    sink.send(Message::Text(text)).await.map_err(|e| format!("sending frame: {e}"))
}
