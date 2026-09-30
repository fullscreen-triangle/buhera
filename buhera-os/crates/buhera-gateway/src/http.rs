//! The HTTP surface.
//!
//! Routes divide into three groups:
//!
//! * `/api/auth/login`  — the only unauthenticated route. There is no public
//!   signup: accounts are seeded on the host with `buhera-gateway --seed`
//!   (see `main.rs`). A fixed roster, not an open registration surface.
//! * `/api/catalysts/*` — the account's machine roster, session-authenticated,
//!   except `/api/catalysts/whoami` which is catalyst-token-authenticated —
//!   the one route a *machine* calls about itself, used by `buhera-pair`.
//! * `/api/run`, `/api/dispatch` — submit work; private to the caller's
//!   account unless `/api/dispatch` names an `experiment`, in which case
//!   the caller must hold standing on it (spec-free, this crate's own:
//!   see `/api/experiments/*` and `store::Experiment`).
//! * `/api/experiments/*` — shared dispatch scopes. The owner creates one
//!   and grants other accounts a capped roster of capabilities on it;
//!   every grantee's `/api/dispatch` calls against it share one module
//!   federation and one audit log, so account identity there is
//!   provenance on an act, never a partition of what the experiment
//!   contains.
//! * `/api/profiles` — a cross-account directory (every account, its
//!   catalysts, its experiment standings), gated by its own shared-secret
//!   scheme (`profiles_token`), not by any account's session token. See
//!   that route's own docs for why.
//!
//! Every authenticated route resolves its account from a *verified* token
//! (see [`crate::token`]), never from a header the caller can simply assert.
//!
//! Failure responses are deliberately uninformative: every authentication
//! failure is the same 401 with the same body, so the API cannot be used to
//! learn whether an address is registered or whether a token was merely
//! expired rather than forged. The distinction is kept in the logs.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

use crate::router::{route, FallbackReason, Route};
use crate::store::{Store, StoreError};
use crate::token::{Audience, Signer};
use crate::{now_unix, session};

/// How long a browser session token is good for.
const SESSION_TTL_SECS: i64 = 12 * 3600;

/// How long a catalyst registration token is good for.
///
/// Much longer than a session: the point of pairing a machine once is that
/// it keeps working. It is still bounded, so a token copied off a decommissioned
/// laptop does not grant access forever.
const CATALYST_TTL_SECS: i64 = 90 * 24 * 3600;

/// Shared server state.
pub struct AppState {
    /// Durable accounts and catalyst roster.
    ///
    /// A mutex rather than a pool: SQLite serialises writers anyway, and the
    /// gateway's account traffic is low — login and pairing, not per-keystroke.
    /// The kernel work that *is* hot does not touch this.
    pub store: Mutex<Store>,
    /// Token signer.
    pub signer: Signer,
    /// Per-account kernels for the degraded path.
    pub sessions: session::Sessions,
    /// Live catalyst connections, dialed out by the machine itself.
    pub relay: crate::relay::Relay,
    /// Per-account module federations for `/api/dispatch` (spec 07 §2):
    /// module state (a vaHera kernel, a pylon-free Rust module set) is scoped
    /// to the account, exactly as `sessions` scopes the `/api/run` kernel.
    pub federations: std::sync::Mutex<std::collections::HashMap<String, buhera_registry::Registry>>,
    /// Per-experiment module federations, shared by every account holding a
    /// grant on that experiment — the collaboration surface the account-keyed
    /// `federations` map deliberately does not provide. Keyed by experiment
    /// id, not by any account, so two grantees dispatching into the same
    /// experiment see the same state and the same audit log.
    pub experiment_federations: std::sync::Mutex<std::collections::HashMap<String, buhera_registry::Registry>>,
    /// Shared secret gating `GET /api/profiles` (see that handler's docs).
    /// `None` — the default, and what a fresh `AppState::new` gets — means
    /// the route refuses every request; there is no "no token configured
    /// so it's open" fallback. Set with [`AppState::with_profiles_token`].
    pub profiles_token: Option<String>,
}

/// The gateway's federation for one account. Filesystem-reading operations are
/// off: the gateway has no business reading its own disk on a caller's behalf
/// (spec 07, rule B5).
fn account_federation() -> buhera_registry::Registry {
    buhera_modules::federation(buhera_modules::Options { filesystem: false }).0
}

impl AppState {
    /// Build the shared state.
    pub fn new(store: Store, signer: Signer) -> Self {
        Self {
            store: Mutex::new(store),
            signer,
            sessions: session::Sessions::new(),
            relay: crate::relay::Relay::new(),
            federations: std::sync::Mutex::new(std::collections::HashMap::new()),
            experiment_federations: std::sync::Mutex::new(std::collections::HashMap::new()),
            profiles_token: None,
        }
    }

    /// Enable `GET /api/profiles`, gated by this shared secret.
    pub fn with_profiles_token(mut self, token: Option<String>) -> Self {
        self.profiles_token = token;
        self
    }
}

/// The API error type.
///
/// `Unauthorized` deliberately carries no detail — see the module docs.
#[derive(Debug)]
pub enum ApiError {
    /// Credentials absent, malformed, expired, or forged.
    Unauthorized,
    /// Well-formed request the server refuses.
    BadRequest(String),
    /// Address already registered.
    Conflict(String),
    /// Named thing does not exist.
    NotFound(String),
    /// Work could not be placed anywhere.
    Unroutable(String),
    /// Anything unexpected.
    Internal(String),
}

impl From<StoreError> for ApiError {
    fn from(e: StoreError) -> Self {
        match e {
            StoreError::Duplicate => ApiError::Conflict("already registered".into()),
            StoreError::NoSuchAccount => ApiError::Unauthorized,
            StoreError::NoSuchExperiment => ApiError::NotFound("no such experiment".into()),
            other => ApiError::Internal(other.to_string()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            ApiError::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized".to_string()),
            ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            ApiError::Conflict(m) => (StatusCode::CONFLICT, m),
            ApiError::NotFound(m) => (StatusCode::NOT_FOUND, m),
            ApiError::Unroutable(m) => (StatusCode::SERVICE_UNAVAILABLE, m),
            ApiError::Internal(m) => {
                // Log the detail, return none of it.
                tracing::error!(error = %m, "internal error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal error".to_string(),
                )
            }
        };
        (status, Json(serde_json::json!({ "ok": false, "error": msg }))).into_response()
    }
}

/// Pull a bearer token out of the `Authorization` header.
fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// Resolve the calling account from a verified session token.
///
/// Returns the account id. Any failure — missing, malformed, expired,
/// forged, or minted for a machine rather than a browser — is the same
/// `Unauthorized`.
fn authenticate(state: &AppState, headers: &HeaderMap) -> Result<String, ApiError> {
    let token = bearer(headers).ok_or(ApiError::Unauthorized)?;
    let claims = state
        .signer
        .verify(token, Audience::Session, now_unix())
        .map_err(|e| {
            tracing::debug!(reason = %e, "session token rejected");
            ApiError::Unauthorized
        })?;
    Ok(claims.subject)
}

/// Resolve the calling account from a verified catalyst token.
///
/// The machine-side counterpart of [`authenticate`] — same shape, different
/// audience. Used by routes a paired machine calls about itself, not by the
/// browser.
fn authenticate_catalyst(state: &AppState, headers: &HeaderMap) -> Result<String, ApiError> {
    let token = bearer(headers).ok_or(ApiError::Unauthorized)?;
    let claims = state
        .signer
        .verify(token, Audience::Catalyst, now_unix())
        .map_err(|e| {
            tracing::debug!(reason = %e, "catalyst token rejected");
            ApiError::Unauthorized
        })?;
    Ok(claims.subject)
}

// ─────────────────────────── auth ───────────────────────────

/// Login request body.
#[derive(Debug, Deserialize)]
pub struct Credentials {
    /// Login address.
    pub email: String,
    /// Plaintext password, over TLS, never logged.
    pub password: String,
}

/// A minted session.
#[derive(Debug, Serialize)]
pub struct SessionResponse {
    /// Always true on this path.
    pub ok: bool,
    /// The bearer token the browser presents.
    pub token: String,
    /// Account id.
    pub account_id: String,
    /// Unix seconds at which `token` stops being accepted.
    pub expires_at: i64,
}

/// The shortest password accepted.
///
/// Low bars invite reuse of a throwaway; a very high bar invites writing it
/// down. Twelve with no composition rules is the current consensus. Also
/// enforced by the `--seed` path in `main.rs`, since that is now the only
/// way an account is created.
pub const MIN_PASSWORD_LEN: usize = 12;

async fn login(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Credentials>,
) -> Result<Json<SessionResponse>, ApiError> {
    let now = now_unix();
    let account = {
        let store = state.store.lock().await;
        store.verify_login(&body.email, &body.password)?
    }
    .ok_or(ApiError::Unauthorized)?;

    let token = state
        .signer
        .mint(Audience::Session, &account.id, now, SESSION_TTL_SECS);
    Ok(Json(SessionResponse {
        ok: true,
        token,
        account_id: account.id,
        expires_at: now + SESSION_TTL_SECS,
    }))
}

// ───────────────────────── catalysts ─────────────────────────

/// One machine, as the API renders it.
#[derive(Debug, Serialize)]
pub struct CatalystView {
    /// Name, unique within the account.
    pub name: String,
    /// Advertised capability tags.
    pub capabilities: Vec<String>,
    /// Whether it is connected right now — the awake/asleep signal.
    pub live: bool,
    /// Last heartbeat, Unix seconds.
    pub last_seen: Option<i64>,
}

async fn list_catalysts(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    let account_id = authenticate(&state, &headers)?;
    let now = now_unix();
    let store = state.store.lock().await;
    let items: Vec<CatalystView> = store
        .catalysts(&account_id)?
        .into_iter()
        .map(|c| CatalystView {
            live: crate::router::is_live(&c, now),
            name: c.name,
            capabilities: c.capabilities,
            last_seen: c.last_seen,
        })
        .collect();
    Ok(Json(serde_json::json!({ "ok": true, "catalysts": items })))
}

/// Request to pair a machine.
#[derive(Debug, Deserialize)]
pub struct PairRequest {
    /// A name for the machine, e.g. "office".
    pub name: String,
    /// What it can do, e.g. `["kernel", "spraypaint"]`.
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// The credential a freshly paired machine is given.
#[derive(Debug, Serialize)]
pub struct PairResponse {
    /// Always true on this path.
    pub ok: bool,
    /// The machine's name.
    pub name: String,
    /// The catalyst token. Shown **once** — the gateway stores no copy it
    /// could show again, because it keeps no plaintext credential.
    pub token: String,
    /// Unix seconds at which the token stops being accepted.
    pub expires_at: i64,
}

async fn pair_catalyst(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<PairRequest>,
) -> Result<Json<PairResponse>, ApiError> {
    let account_id = authenticate(&state, &headers)?;
    let name = body.name.trim();
    if name.is_empty() {
        return Err(ApiError::BadRequest("machine name required".into()));
    }
    if name == "gateway" {
        // Reserved: the router treats it as "run here", so a machine by
        // that name could never be selected and would confuse the roster.
        return Err(ApiError::BadRequest(
            "\"gateway\" is reserved; choose another name".into(),
        ));
    }

    let now = now_unix();
    {
        let store = state.store.lock().await;
        store.upsert_catalyst(&account_id, name, &body.capabilities, now)?;
    }

    let token = state
        .signer
        .mint(Audience::Catalyst, &account_id, now, CATALYST_TTL_SECS);
    Ok(Json(PairResponse {
        ok: true,
        name: name.to_string(),
        token,
        expires_at: now + CATALYST_TTL_SECS,
    }))
}

/// What a machine learns about itself by presenting its catalyst token.
///
/// Deliberately minimal — the account id and nothing else. A paired
/// machine does not need to see its sibling machines or the account's
/// email; it only needs to confirm the token it was given still works, for
/// `buhera-pair status`.
#[derive(Debug, Serialize)]
pub struct CatalystWhoami {
    /// Always true on this path.
    pub ok: bool,
    /// The account this catalyst token belongs to.
    pub account_id: String,
}

async fn catalyst_whoami(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<CatalystWhoami>, ApiError> {
    let account_id = authenticate_catalyst(&state, &headers)?;
    Ok(Json(CatalystWhoami { ok: true, account_id }))
}

async fn unpair_catalyst(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let account_id = authenticate(&state, &headers)?;
    let store = state.store.lock().await;
    if store.remove_catalyst(&account_id, &name)? {
        Ok(Json(serde_json::json!({ "ok": true })))
    } else {
        Err(ApiError::NotFound(format!("no machine named {name:?}")))
    }
}

/// `GET /api/catalysts/relay` — the connection a paired machine dials out
/// over. Catalyst-token-authenticated, like `/api/catalysts/whoami`; unlike
/// it, this connection is held open for as long as the machine wants to be
/// dispatchable.
async fn catalyst_relay(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ws: axum::extract::WebSocketUpgrade,
) -> Result<Response, ApiError> {
    let account_id = authenticate_catalyst(&state, &headers)?;
    Ok(ws.on_upgrade(move |socket| relay_connection(state, account_id, socket)))
}

/// Drive one catalyst's connection for as long as it stays open.
///
/// The first frame must be `Hello`, naming which of the account's paired
/// machines this is — the catalyst token alone does not disambiguate, since
/// one account can hold several. Anything else first, or a name that is not
/// actually paired to this account, closes the socket without registering
/// it: a connection that never announces itself correctly can never be
/// dispatched to, so there is nothing to clean up by leaving it open.
async fn relay_connection(state: Arc<AppState>, account_id: String, mut socket: axum::extract::ws::WebSocket) {
    use axum::extract::ws::Message;
    use futures_util::SinkExt;

    let name = match socket.recv().await {
        Some(Ok(Message::Text(text))) => match serde_json::from_str::<crate::relay::ClientMsg>(&text) {
            Ok(crate::relay::ClientMsg::Hello { name }) => name,
            _ => {
                tracing::debug!("relay connection's first frame was not Hello; closing");
                return;
            }
        },
        _ => return,
    };

    let known = {
        match state.store.lock().await.catalysts(&account_id) {
            Ok(catalysts) => catalysts.iter().any(|c| c.name == name),
            Err(e) => {
                tracing::error!(error = %e, "could not verify catalyst roster for relay connection");
                false
            }
        }
    };
    if !known {
        tracing::debug!(%account_id, %name, "relay Hello named a machine not paired to this account; closing");
        return;
    }

    let (mut sink, mut stream) = socket.split();
    let (outbound_tx, mut outbound_rx) = tokio::sync::mpsc::unbounded_channel::<crate::relay::ServerMsg>();

    state.relay.register(account_id.clone(), name.clone(), outbound_tx.clone());
    let now = now_unix();
    if let Err(e) = state.store.lock().await.touch_catalyst(&account_id, &name, now) {
        tracing::warn!(error = %e, %account_id, %name, "could not record initial relay heartbeat");
    }
    tracing::info!(%account_id, %name, "catalyst connected");

    let ack = serde_json::to_string(&crate::relay::ServerMsg::HelloAck).expect("HelloAck serializes");
    if sink.send(Message::Text(ack)).await.is_err() {
        state.relay.deregister(&account_id, &name, &outbound_tx);
        return;
    }

    // Forward outbound frames (dispatched runs) to the socket, and read
    // inbound frames (heartbeats and run answers), until either side closes.
    let writer_account = account_id.clone();
    let writer_name = name.clone();
    let writer = tokio::spawn(async move {
        while let Some(msg) = outbound_rx.recv().await {
            let text = match serde_json::to_string(&msg) {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(error = %e, "could not serialize a relay ServerMsg");
                    continue;
                }
            };
            if sink.send(Message::Text(text)).await.is_err() {
                break;
            }
        }
        let _ = (writer_account, writer_name);
    });

    use futures_util::StreamExt;
    while let Some(frame) = stream.next().await {
        let text = match frame {
            Ok(Message::Text(t)) => t,
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(_) => continue,
        };
        let msg: crate::relay::ClientMsg = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!(error = %e, "unparseable relay frame; ignoring");
                continue;
            }
        };
        if let crate::relay::ClientMsg::Heartbeat = msg {
            let now = now_unix();
            if let Err(e) = state.store.lock().await.touch_catalyst(&account_id, &name, now) {
                tracing::warn!(error = %e, %account_id, %name, "could not record relay heartbeat");
            }
            continue;
        }
        crate::relay::resolve_client_msg(&state.relay, &account_id, &name, msg);
    }

    writer.abort();
    state.relay.deregister(&account_id, &name, &outbound_tx);
    tracing::info!(%account_id, %name, "catalyst disconnected");
}

// ─────────────────────────── run ───────────────────────────

/// A unit of work.
#[derive(Debug, Deserialize)]
pub struct RunRequest {
    /// vaHera source to execute.
    pub source: String,
    /// What executing it needs. Defaults to `vahera`, which the gateway
    /// can serve itself.
    #[serde(default = "default_capability")]
    pub capability: String,
    /// Optionally pin a machine by name, or `"gateway"` to force local.
    #[serde(default)]
    pub prefer: Option<String>,
}

fn default_capability() -> String {
    "vahera".to_string()
}

/// Result of a run, including where it ran.
#[derive(Debug, Serialize)]
pub struct RunResponse {
    /// Always true on this path.
    pub ok: bool,
    /// `"gateway"` or the machine's name.
    pub executed_on: String,
    /// Present when the gateway stood in for a machine — the user should
    /// be told their session is running degraded rather than quietly
    /// getting different answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Rendered results, one entry per statement that produced output.
    pub results: Vec<serde_json::Value>,
    /// Interpreter trace.
    pub trace: Vec<String>,
}

async fn run(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<RunRequest>,
) -> Result<Json<RunResponse>, ApiError> {
    let account_id = authenticate(&state, &headers)?;
    let now = now_unix();

    let catalysts = {
        let store = state.store.lock().await;
        store.catalysts(&account_id)?
    };

    let decision = route(&catalysts, &body.capability, body.prefer.as_deref(), now)
        .map_err(|e| ApiError::Unroutable(e.to_string()))?;

    match decision {
        Route::Gateway { reason } => {
            let out = state
                .sessions
                .execute(&account_id, &body.source)
                .map_err(|e| ApiError::BadRequest(e.to_string()))?;
            Ok(Json(RunResponse {
                ok: true,
                executed_on: "gateway".to_string(),
                note: match reason {
                    // Not worth a banner when the user asked for it.
                    FallbackReason::Requested => None,
                    other => Some(other.describe().to_string()),
                },
                results: out.results,
                trace: out.trace,
            }))
        }
        Route::Catalyst { name } => {
            let outcome = state.relay.dispatch(&account_id, &name, &body.source).await.map_err(|e| {
                // The router saw this catalyst as live a moment ago (its
                // last heartbeat was inside the liveness window), but the
                // socket itself is the source of truth for whether work can
                // actually reach it — the two can disagree by a few seconds.
                // Report it plainly rather than silently falling back to
                // the gateway, which would answer against the wrong
                // filesystem.
                ApiError::Unroutable(format!(
                    "machine {name:?} could not run this: {e}; retry with prefer=\"gateway\" to run on the server"
                ))
            })?;
            Ok(Json(RunResponse {
                ok: true,
                executed_on: name,
                note: None,
                results: outcome.results,
                trace: outcome.trace,
            }))
        }
    }
}

// ─────────────────────────── experiments ───────────────────────────
//
// An experiment is a shared dispatch scope: one account owns it, any
// number of other accounts can hold a grant on it, and every grantee
// dispatching against it sees the same module federation and the same
// audit log — the collaboration surface `/api/dispatch`'s per-account
// federation deliberately does not provide (see `store::Experiment`).
//
// What an account's identity does here is exactly provenance: every
// audited act names the `account_id` that made it, and nothing about the
// shared state is ever filtered, gated, or partitioned by that id. A
// grant narrows *capability* (which modules an account may dispatch),
// never *visibility* of what others in the same experiment have already
// done — the audit log is the one shared, append-only record everyone
// with standing can read in full.

/// Body of `POST /api/experiments`.
#[derive(Debug, Deserialize)]
pub struct CreateExperimentRequest {
    /// Human label. Not unique; display only.
    pub name: String,
}

/// One experiment as the API renders it.
#[derive(Debug, Serialize)]
pub struct ExperimentView {
    /// Stable opaque id.
    pub id: String,
    /// The account that owns it and can grant/revoke access.
    pub owner_account_id: String,
    /// Human label.
    pub name: String,
    /// Unix seconds at creation.
    pub created_at: i64,
    /// This caller's standing: `"owner"` or the capability list of its grant.
    pub standing: StandingView,
}

/// JSON-friendly rendering of [`crate::store::ExperimentStanding`].
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StandingView {
    /// Every capability, plus the right to grant and revoke.
    Owner,
    /// Capped to these capabilities.
    Grantee {
        /// The capability tags this account may dispatch.
        capabilities: Vec<String>,
    },
}

impl From<crate::store::ExperimentStanding> for StandingView {
    fn from(s: crate::store::ExperimentStanding) -> Self {
        match s {
            crate::store::ExperimentStanding::Owner => StandingView::Owner,
            crate::store::ExperimentStanding::Grantee(capabilities) => StandingView::Grantee { capabilities },
        }
    }
}

async fn create_experiment(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<CreateExperimentRequest>,
) -> Result<Json<ExperimentView>, ApiError> {
    let account_id = authenticate(&state, &headers)?;
    let name = body.name.trim();
    if name.is_empty() {
        return Err(ApiError::BadRequest("name is required".into()));
    }
    let now = now_unix();
    let exp = {
        let store = state.store.lock().await;
        store.create_experiment(&account_id, name, now)?
    };
    Ok(Json(ExperimentView {
        id: exp.id,
        owner_account_id: exp.owner_account_id,
        name: exp.name,
        created_at: exp.created_at,
        standing: StandingView::Owner,
    }))
}

/// `GET /api/experiments` — every experiment the caller owns or is granted into.
async fn list_experiments(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    let account_id = authenticate(&state, &headers)?;
    let store = state.store.lock().await;
    let experiments = store.experiments_for_account(&account_id)?;
    let mut views = Vec::with_capacity(experiments.len());
    for exp in experiments {
        // Each of these accounts is exactly the ones just listed as owned
        // or granted, so a standing always exists here.
        let standing = store
            .experiment_grant_for(&exp.id, &account_id)?
            .expect("account_id was just listed as having standing on this experiment");
        views.push(ExperimentView {
            id: exp.id,
            owner_account_id: exp.owner_account_id,
            name: exp.name,
            created_at: exp.created_at,
            standing: standing.into(),
        });
    }
    Ok(Json(serde_json::json!({ "ok": true, "experiments": views })))
}

/// Body of `POST /api/experiments/:id/grants`.
#[derive(Debug, Deserialize)]
pub struct GrantRequest {
    /// The account to grant, by email — matching how a person identifies
    /// a collaborator, not by an opaque id they would have to ask for.
    pub email: String,
    /// Capability tags this account may dispatch inside the experiment.
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// One entry in an experiment's roster, as the owner reviews it.
#[derive(Debug, Serialize)]
pub struct GrantView {
    /// The granted account.
    pub account_id: String,
    /// The capability tags this account may dispatch.
    pub capabilities: Vec<String>,
    /// Unix seconds when the grant was made (or last replaced).
    pub granted_at: i64,
}

async fn grant(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(experiment_id): Path<String>,
    Json(body): Json<GrantRequest>,
) -> Result<Json<GrantView>, ApiError> {
    let account_id = authenticate(&state, &headers)?;
    let now = now_unix();
    let store = state.store.lock().await;
    require_owner(&store, &experiment_id, &account_id)?;

    let grantee_email = body.email.trim().to_lowercase();
    if grantee_email.is_empty() {
        return Err(ApiError::BadRequest("email is required".into()));
    }
    // No account enumeration via this route either: a caller who already
    // holds ownership of the experiment can still only learn "no such
    // account", never distinguish that from any other rejection reason.
    let grantee = store
        .account_by_email(&grantee_email)?
        .ok_or_else(|| ApiError::NotFound("no account with that email".into()))?;

    let g = store.grant_experiment(&experiment_id, &grantee.id, &body.capabilities, now)?;
    Ok(Json(GrantView { account_id: g.account_id, capabilities: g.capabilities, granted_at: g.granted_at }))
}

/// `GET /api/experiments/:id/grants` — the owner's view of the roster.
async fn list_grants(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(experiment_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let account_id = authenticate(&state, &headers)?;
    let store = state.store.lock().await;
    require_owner(&store, &experiment_id, &account_id)?;
    let grants: Vec<GrantView> = store
        .experiment_grants(&experiment_id)?
        .into_iter()
        .map(|g| GrantView { account_id: g.account_id, capabilities: g.capabilities, granted_at: g.granted_at })
        .collect();
    Ok(Json(serde_json::json!({ "ok": true, "grants": grants })))
}

async fn revoke_grant(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path((experiment_id, grantee_account_id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let account_id = authenticate(&state, &headers)?;
    let store = state.store.lock().await;
    require_owner(&store, &experiment_id, &account_id)?;
    if store.revoke_experiment_grant(&experiment_id, &grantee_account_id)? {
        Ok(Json(serde_json::json!({ "ok": true })))
    } else {
        Err(ApiError::NotFound("no such grant".into()))
    }
}

/// Confirm `account_id` owns `experiment_id`, mapping every other
/// standing (grantee or none) to the same `Unauthorized` — a grantee
/// probing whether they can manage the roster learns nothing about
/// whether the experiment exists that `/api/experiments` wouldn't
/// already have told them.
fn require_owner(store: &Store, experiment_id: &str, account_id: &str) -> Result<(), ApiError> {
    match store.experiment_grant_for(experiment_id, account_id)? {
        Some(crate::store::ExperimentStanding::Owner) => Ok(()),
        _ => Err(ApiError::Unauthorized),
    }
}

// ─────────────────────────── profiles ───────────────────────────
//
// A read-only, cross-account directory: every seeded account, its
// catalysts, and its experiment standings — composed entirely from reads
// `store.rs` already exposes for other routes. Nothing here is reachable
// by a session token: no account, however privileged its own data, is
// entitled to enumerate every other account's, and there is no role
// system in this crate to grant such an entitlement narrowly. Instead
// this route has its own, separate shared-secret scheme, deliberately
// independent of `Signer`/`Audience` — the intended caller is a build or
// tooling process with no account of its own, not a browser session. The
// pattern mirrors `long-grass`'s own `BUHERA_DISPATCH_TOKEN` on
// `pages/api/dispatch.js`.

/// One account as `/api/profiles` renders it.
#[derive(Debug, Serialize)]
pub struct ProfileView {
    /// Stable opaque id.
    pub account_id: String,
    /// Login address.
    pub email: String,
    /// Unix seconds at creation.
    pub created_at: i64,
    /// This account's paired machines.
    pub catalysts: Vec<CatalystView>,
    /// Experiments this account owns.
    pub experiments_owned: Vec<OwnedExperimentView>,
    /// Experiments this account holds a grant on (not counting the ones
    /// it owns, which are listed separately above).
    pub experiments_granted: Vec<GrantedExperimentView>,
}

/// One owned experiment, from the owner's side.
#[derive(Debug, Serialize)]
pub struct OwnedExperimentView {
    /// Stable opaque id.
    pub id: String,
    /// Human label.
    pub name: String,
    /// Unix seconds at creation.
    pub created_at: i64,
    /// How many accounts hold a grant on it (not counting the owner).
    pub grantee_count: usize,
}

/// One granted experiment, from the grantee's side.
#[derive(Debug, Serialize)]
pub struct GrantedExperimentView {
    /// Stable opaque id.
    pub id: String,
    /// Human label.
    pub name: String,
    /// The account that owns it.
    pub owner_account_id: String,
    /// The capability tags this account may dispatch inside it.
    pub capabilities: Vec<String>,
}

/// Confirm the caller presented the configured `profiles_token` exactly.
/// A missing configuration refuses every request rather than defaulting
/// open — see [`AppState::profiles_token`].
fn authenticate_profiles(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(expected) = &state.profiles_token else {
        return Err(ApiError::Unauthorized);
    };
    let presented = bearer(headers).ok_or(ApiError::Unauthorized)?;
    // Constant-time compare: this is a long-lived shared secret, not a
    // per-account, expiring, signed token like everything else in this
    // file, so nothing else here already protects it from timing analysis.
    use subtle::ConstantTimeEq;
    if presented.as_bytes().ct_eq(expected.as_bytes()).unwrap_u8() != 1 {
        return Err(ApiError::Unauthorized);
    }
    Ok(())
}

/// `GET /api/profiles` — every account, its machines, and its experiment
/// standings. See the module docs above for why this is not
/// session-authenticated.
async fn profiles(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    authenticate_profiles(&state, &headers)?;
    let now = now_unix();
    let store = state.store.lock().await;

    let mut views = Vec::new();
    for account in store.accounts()? {
        let catalysts: Vec<CatalystView> = store
            .catalysts(&account.id)?
            .into_iter()
            .map(|c| CatalystView { live: crate::router::is_live(&c, now), name: c.name, capabilities: c.capabilities, last_seen: c.last_seen })
            .collect();

        let mut experiments_owned = Vec::new();
        let mut experiments_granted = Vec::new();
        for exp in store.experiments_for_account(&account.id)? {
            if exp.owner_account_id == account.id {
                let grantee_count = store.experiment_grants(&exp.id)?.len();
                experiments_owned.push(OwnedExperimentView { id: exp.id, name: exp.name, created_at: exp.created_at, grantee_count });
            } else {
                let crate::store::ExperimentStanding::Grantee(capabilities) =
                    store.experiment_grant_for(&exp.id, &account.id)?.ok_or(ApiError::Internal(
                        "experiments_for_account listed an experiment with no standing".into(),
                    ))?
                else {
                    // experiments_for_account only lists owned-or-granted, and the
                    // owned case was already handled above.
                    return Err(ApiError::Internal("owner standing on a non-owned experiment".into()));
                };
                experiments_granted.push(GrantedExperimentView {
                    id: exp.id,
                    name: exp.name,
                    owner_account_id: exp.owner_account_id,
                    capabilities,
                });
            }
        }

        views.push(ProfileView {
            account_id: account.id,
            email: account.email,
            created_at: account.created_at,
            catalysts,
            experiments_owned,
            experiments_granted,
        });
    }

    Ok(Json(serde_json::json!({ "ok": true, "profiles": views })))
}

// ─────────────────────────── module dispatch ───────────────────────────

/// Body of `POST /api/dispatch` (specification 07 §2.1).
#[derive(Debug, Deserialize)]
pub struct DispatchRequest {
    /// Registry module id.
    pub module: String,
    /// The instruction, verbatim.
    #[serde(default)]
    pub instruction: serde_json::Value,
    /// Act budget (≥ 1).
    #[serde(default = "one")]
    pub act_budget: u32,
    /// Dispatch into a shared experiment's federation instead of the
    /// caller's private one. The caller must hold standing on it, and if
    /// they are a grantee (not the owner) `module` must be within their
    /// grant's capabilities — a ceiling on what a grant can reach, never
    /// a way around what the account could already do unaided.
    #[serde(default)]
    pub experiment: Option<String>,
}

fn one() -> u32 {
    1
}

/// Reply of `POST /api/dispatch` (specification 07 §2.2).
#[derive(Debug, Serialize)]
pub struct DispatchResponse {
    /// Always `"gateway"` until the catalyst relay exists.
    pub executed_on: String,
    /// The module's ActResult, verbatim — `ok:false` is the module's verdict,
    /// not a transport failure.
    pub result: buhera_registry::ActResult,
    /// The act id in the audit log this act was recorded to — the
    /// caller's own when `experiment` was absent, the experiment's shared
    /// log otherwise.
    pub act_id: u64,
}

async fn dispatch(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<DispatchRequest>,
) -> Result<Json<DispatchResponse>, ApiError> {
    let account_id = authenticate(&state, &headers)?;

    let experiment_id = if let Some(experiment_id) = &body.experiment {
        let store = state.store.lock().await;
        let standing = store
            .experiment_grant_for(experiment_id, &account_id)?
            .ok_or(ApiError::Unauthorized)?;
        if !standing.allows(&body.module) {
            return Err(ApiError::Unauthorized);
        }
        Some(experiment_id.clone())
    } else {
        None
    };

    // Module acts are synchronous CPU work (contract M5); run them off the
    // async executor.
    tokio::task::block_in_place(|| {
        let (result, act_id) = if let Some(experiment_id) = experiment_id {
            let mut feds = state
                .experiment_federations
                .lock()
                .map_err(|_| ApiError::Internal("experiment federation lock poisoned".into()))?;
            let reg = feds.entry(experiment_id).or_insert_with(account_federation);
            let result = reg
                .dispatch(&body.module, body.instruction, body.act_budget.max(1))
                .map_err(|e| ApiError::NotFound(e.to_string()))?;
            let act_id = reg.audit_log().last().map(|e| e.act_id).unwrap_or(0);
            (result, act_id)
        } else {
            let mut feds =
                state.federations.lock().map_err(|_| ApiError::Internal("federation lock poisoned".into()))?;
            let reg = feds.entry(account_id).or_insert_with(account_federation);
            let result = reg
                .dispatch(&body.module, body.instruction, body.act_budget.max(1))
                .map_err(|e| ApiError::NotFound(e.to_string()))?;
            let act_id = reg.audit_log().last().map(|e| e.act_id).unwrap_or(0);
            (result, act_id)
        };
        Ok(Json(DispatchResponse { executed_on: "gateway".into(), result, act_id }))
    })
}

/// `GET /api/modules` — what `/api/dispatch` can reach.
async fn list_modules(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    authenticate(&state, &headers)?;
    let (modules, dsls) = buhera_modules::federation(buhera_modules::Options { filesystem: false });
    Ok(Json(serde_json::json!({ "modules": modules.list(), "dsls": dsls.list() })))
}

// ─────────────────────────── wiring ───────────────────────────

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true, "service": "buhera-gateway" }))
}

/// Build the router.
///
/// CORS is wide open (`Any` origin/method/header): every route here either
/// takes no credentials (`/api/auth/*`) or is authenticated by a bearer token
/// the caller must attach itself, never by a cookie the browser would send
/// automatically — so there is no ambient credential for a foreign origin to
/// ride in on, and restricting `Origin` would only block legitimate browser
/// clients (long-grass and anything else built against this API) without
/// stopping anything a curious server-side caller couldn't already do.
pub fn app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/auth/login", post(login))
        .route("/api/catalysts", get(list_catalysts).post(pair_catalyst))
        .route("/api/catalysts/whoami", get(catalyst_whoami))
        .route("/api/catalysts/relay", get(catalyst_relay))
        .route("/api/catalysts/:name", axum::routing::delete(unpair_catalyst))
        .route("/api/run", post(run))
        .route("/api/experiments", get(list_experiments).post(create_experiment))
        .route("/api/experiments/:id/grants", get(list_grants).post(grant))
        .route("/api/experiments/:id/grants/:account_id", axum::routing::delete(revoke_grant))
        .route("/api/dispatch", post(dispatch))
        .route("/api/modules", get(list_modules))
        .route("/api/profiles", get(profiles))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .with_state(state)
}

#[cfg(test)]
mod dispatch_tests {
    //! `/api/dispatch` (specification 07 §2): authentication, R1 over HTTP,
    //! a real act, and per-account state.
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use tower::ServiceExt;

    fn state() -> (Arc<AppState>, String, String) {
        let signer = Signer::new(vec![7u8; 32]).unwrap();
        let alice = signer.mint(Audience::Session, "alice", now_unix(), 3600);
        let bob = signer.mint(Audience::Session, "bob", now_unix(), 3600);
        (Arc::new(AppState::new(Store::open_memory().unwrap(), signer)), alice, bob)
    }

    async fn post(state: &Arc<AppState>, token: Option<&str>, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
        let mut req = Request::post("/api/dispatch").header("content-type", "application/json");
        if let Some(t) = token {
            req = req.header("authorization", format!("Bearer {t}"));
        }
        let res = app(state.clone()).oneshot(req.body(Body::from(body.to_string())).unwrap()).await.unwrap();
        let status = res.status();
        let bytes = to_bytes(res.into_body(), 1 << 22).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unauthenticated_dispatch_is_refused() {
        let (s, _, _) = state();
        let (status, _) = post(&s, None, serde_json::json!({ "module": "ndombolo", "instruction": "demo" })).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unknown_module_is_404() {
        let (s, alice, _) = state();
        let (status, _) = post(&s, Some(&alice), serde_json::json!({ "module": "nope" })).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_real_act_runs_and_is_audited_per_account() {
        let (s, alice, bob) = state();
        let (status, body) = post(&s, Some(&alice), serde_json::json!({ "module": "ndombolo", "instruction": "demo" })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["executed_on"], "gateway");
        assert_eq!(body["result"]["ok"], true);
        assert_eq!(body["result"]["output_delta"]["kind"], "ndombolo_result");
        assert_eq!(body["act_id"], 1);

        // vaHera kernel state is per account (B4).
        post(&s, Some(&alice), serde_json::json!({ "module": "vahera", "instruction": "memory store \"k\" = \"alice only\"" })).await;
        let (_, a) = post(&s, Some(&alice), serde_json::json!({ "module": "vahera", "instruction": "memory list" })).await;
        let (_, b) = post(&s, Some(&bob), serde_json::json!({ "module": "vahera", "instruction": "memory list" })).await;
        assert!(a.to_string().contains("\"k\""));
        assert!(!b.to_string().contains("\"k\""));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn filesystem_operations_are_off_on_the_gateway() {
        let (s, alice, _) = state();
        let (_, body) = post(&s, Some(&alice), serde_json::json!({ "module": "tracker", "instruction": { "kind": "list", "root": "/" } })).await;
        assert_eq!(body["result"]["error"], "unavailable on this host");
    }
}

#[cfg(test)]
mod experiment_tests {
    //! `/api/experiments/*` and the `experiment` field on `/api/dispatch`:
    //! shared federation, capability as a ceiling, and no partitioning of
    //! the shared state by who's calling.
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use tower::ServiceExt;

    /// Same shape as `dispatch_tests::state`, but the accounts are real
    /// DB rows (email-addressable) rather than bare token subjects, since
    /// granting works by email.
    async fn state() -> (Arc<AppState>, String, crate::store::Account, crate::store::Account) {
        let signer = Signer::new(vec![7u8; 32]).unwrap();
        let store = Store::open_memory().unwrap();
        let now = now_unix();
        let alice = store.create_account("alice@example.org", "correct horse battery", now).unwrap();
        let bob = store.create_account("bob@example.org", "correct horse battery", now).unwrap();
        let app_state = Arc::new(AppState::new(store, signer));
        (app_state, "unused".into(), alice, bob)
    }

    fn token(state: &AppState, account_id: &str) -> String {
        state.signer.mint(Audience::Session, account_id, now_unix(), 3600)
    }

    async fn call(
        state: &Arc<AppState>,
        method: &str,
        path: &str,
        token: Option<&str>,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, serde_json::Value) {
        let mut req = Request::builder().method(method).uri(path).header("content-type", "application/json");
        if let Some(t) = token {
            req = req.header("authorization", format!("Bearer {t}"));
        }
        let payload = body.map(|b| b.to_string()).unwrap_or_default();
        let res = app(state.clone()).oneshot(req.body(Body::from(payload)).unwrap()).await.unwrap();
        let status = res.status();
        let bytes = to_bytes(res.into_body(), 1 << 22).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn owner_creates_and_sees_it_without_granting_self() {
        let (s, _, alice, _) = state().await;
        let at = token(&s, &alice.id);

        let (status, body) =
            call(&s, "POST", "/api/experiments", Some(&at), Some(serde_json::json!({ "name": "nfdi4cat-run" }))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["standing"]["kind"], "owner");
        let exp_id = body["id"].as_str().unwrap().to_string();

        let (_, list) = call(&s, "GET", "/api/experiments", Some(&at), None).await;
        assert_eq!(list["experiments"].as_array().unwrap().len(), 1);
        assert_eq!(list["experiments"][0]["id"], exp_id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_stranger_cannot_see_or_dispatch_into_an_experiment() {
        let (s, _, alice, bob) = state().await;
        let at = token(&s, &alice.id);
        let bt = token(&s, &bob.id);

        let (_, created) =
            call(&s, "POST", "/api/experiments", Some(&at), Some(serde_json::json!({ "name": "x" }))).await;
        let exp_id = created["id"].as_str().unwrap();

        // Bob has no grant yet — the experiment is invisible to him and
        // dispatch into it is refused, not merely empty.
        let (_, list) = call(&s, "GET", "/api/experiments", Some(&bt), None).await;
        assert_eq!(list["experiments"].as_array().unwrap().len(), 0);

        let (status, _) = call(
            &s,
            "POST",
            "/api/dispatch",
            Some(&bt),
            Some(serde_json::json!({ "module": "ndombolo", "instruction": "demo", "experiment": exp_id })),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn grant_is_a_ceiling_on_capability() {
        let (s, _, alice, bob) = state().await;
        let at = token(&s, &alice.id);
        let bt = token(&s, &bob.id);

        let (_, created) =
            call(&s, "POST", "/api/experiments", Some(&at), Some(serde_json::json!({ "name": "x" }))).await;
        let exp_id = created["id"].as_str().unwrap().to_string();

        // Grant bob only "vahera" — never "ndombolo".
        let (status, _) = call(
            &s,
            "POST",
            &format!("/api/experiments/{exp_id}/grants"),
            Some(&at),
            Some(serde_json::json!({ "email": "bob@example.org", "capabilities": ["vahera"] })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // Now bob can see it and dispatch the granted capability...
        let (_, list) = call(&s, "GET", "/api/experiments", Some(&bt), None).await;
        assert_eq!(list["experiments"][0]["standing"]["kind"], "grantee");

        let (status, body) = call(
            &s,
            "POST",
            "/api/dispatch",
            Some(&bt),
            Some(serde_json::json!({ "module": "vahera", "instruction": "memory list", "experiment": exp_id })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");

        // ...but not one outside the grant, even though ndombolo dispatch
        // succeeds for bob outside any experiment (it is not forbidden to
        // him in general — only inside this experiment's narrower grant).
        let (status, _) = call(
            &s,
            "POST",
            "/api/dispatch",
            Some(&bt),
            Some(serde_json::json!({ "module": "ndombolo", "instruction": "demo", "experiment": exp_id })),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);

        let (status, _) =
            call(&s, "POST", "/api/dispatch", Some(&bt), Some(serde_json::json!({ "module": "ndombolo", "instruction": "demo" })))
                .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn acts_land_in_one_shared_log_not_partitioned_by_account() {
        let (s, _, alice, bob) = state().await;
        let at = token(&s, &alice.id);
        let bt = token(&s, &bob.id);

        let (_, created) =
            call(&s, "POST", "/api/experiments", Some(&at), Some(serde_json::json!({ "name": "x" }))).await;
        let exp_id = created["id"].as_str().unwrap().to_string();
        call(
            &s,
            "POST",
            &format!("/api/experiments/{exp_id}/grants"),
            Some(&at),
            Some(serde_json::json!({ "email": "bob@example.org", "capabilities": ["vahera"] })),
        )
        .await;

        // Alice writes a fact into the shared kernel...
        call(
            &s,
            "POST",
            "/api/dispatch",
            Some(&at),
            Some(serde_json::json!({
                "module": "vahera",
                "instruction": "memory store \"k\" = \"alice wrote this\"",
                "experiment": exp_id,
            })),
        )
        .await;

        // ...and bob, dispatching into the same experiment, sees it. This
        // is the opposite property from `/api/dispatch`'s per-account
        // federation test (`a_real_act_runs_and_is_audited_per_account`),
        // deliberately: inside an experiment, state is shared.
        let (_, b) = call(
            &s,
            "POST",
            "/api/dispatch",
            Some(&bt),
            Some(serde_json::json!({ "module": "vahera", "instruction": "memory list", "experiment": exp_id })),
        )
        .await;
        assert!(b.to_string().contains("\"k\""), "{b}");

        // The act ids are drawn from the experiment's own counter, shared
        // across both accounts' acts — 1 for alice's write, 2 for bob's read.
        assert_eq!(b["act_id"], 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn only_the_owner_manages_the_roster() {
        let (s, _, alice, bob) = state().await;
        let at = token(&s, &alice.id);
        let bt = token(&s, &bob.id);

        let (_, created) =
            call(&s, "POST", "/api/experiments", Some(&at), Some(serde_json::json!({ "name": "x" }))).await;
        let exp_id = created["id"].as_str().unwrap().to_string();

        // Bob, with no standing at all yet, cannot grant himself access.
        let (status, _) = call(
            &s,
            "POST",
            &format!("/api/experiments/{exp_id}/grants"),
            Some(&bt),
            Some(serde_json::json!({ "email": "bob@example.org", "capabilities": ["vahera"] })),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);

        // Alice grants him narrowly...
        call(
            &s,
            "POST",
            &format!("/api/experiments/{exp_id}/grants"),
            Some(&at),
            Some(serde_json::json!({ "email": "bob@example.org", "capabilities": ["vahera"] })),
        )
        .await;

        // ...but even as a grantee, bob still cannot manage the roster —
        // a grant is capability inside the experiment, not co-ownership.
        let (status, _) = call(
            &s,
            "POST",
            &format!("/api/experiments/{exp_id}/grants"),
            Some(&bt),
            Some(serde_json::json!({ "email": "bob@example.org", "capabilities": ["ndombolo"] })),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);

        // Only alice can revoke, and only alice can list the roster.
        let (status, _) = call(&s, "DELETE", &format!("/api/experiments/{exp_id}/grants/{}", bob.id), Some(&bt), None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);

        let (status, grants) = call(&s, "GET", &format!("/api/experiments/{exp_id}/grants"), Some(&at), None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(grants["grants"].as_array().unwrap().len(), 1);

        let (status, _) = call(&s, "DELETE", &format!("/api/experiments/{exp_id}/grants/{}", bob.id), Some(&at), None).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn granting_a_nonexistent_email_is_not_found_not_a_leak() {
        let (s, _, alice, _) = state().await;
        let at = token(&s, &alice.id);
        let (_, created) =
            call(&s, "POST", "/api/experiments", Some(&at), Some(serde_json::json!({ "name": "x" }))).await;
        let exp_id = created["id"].as_str().unwrap().to_string();

        let (status, _) = call(
            &s,
            "POST",
            &format!("/api/experiments/{exp_id}/grants"),
            Some(&at),
            Some(serde_json::json!({ "email": "ghost@example.org", "capabilities": [] })),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}

#[cfg(test)]
mod profiles_tests {
    //! `/api/profiles`: the shared-secret scheme is separate from every
    //! account's session token, refuses by default, and the composed view
    //! matches what the store actually holds.
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use tower::ServiceExt;

    fn signer() -> Signer {
        Signer::new(vec![7u8; 32]).unwrap()
    }

    async fn get(state: &Arc<AppState>, token: Option<&str>) -> (StatusCode, serde_json::Value) {
        let mut req = Request::get("/api/profiles");
        if let Some(t) = token {
            req = req.header("authorization", format!("Bearer {t}"));
        }
        let res = app(state.clone()).oneshot(req.body(Body::empty()).unwrap()).await.unwrap();
        let status = res.status();
        let bytes = to_bytes(res.into_body(), 1 << 22).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn refuses_when_no_token_is_configured() {
        // The important default: absence of configuration is refusal, not
        // an open route. Even a request with no Authorization header at
        // all must not be treated as "any caller is fine".
        let state = Arc::new(AppState::new(Store::open_memory().unwrap(), signer()));
        let (status, _) = get(&state, None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let (status, _) = get(&state, Some("anything")).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn refuses_without_the_configured_token() {
        let state = Arc::new(
            AppState::new(Store::open_memory().unwrap(), signer()).with_profiles_token(Some("secret-1".into())),
        );
        let (status, _) = get(&state, None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let (status, _) = get(&state, Some("wrong")).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_session_token_does_not_work_here() {
        // The whole point of the separate scheme: a real, validly-signed
        // account session must not open this route.
        let s = signer();
        let state = Arc::new(AppState::new(Store::open_memory().unwrap(), s.clone()).with_profiles_token(Some("secret-1".into())));
        let session_token = s.mint(Audience::Session, "some-account-id", now_unix(), 3600);
        let (status, _) = get(&state, Some(&session_token)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn lists_every_account_with_its_catalysts_and_experiments() {
        let store = Store::open_memory().unwrap();
        let now = now_unix();
        let alice = store.create_account("alice@example.org", "correct horse battery", now).unwrap();
        let bob = store.create_account("bob@example.org", "correct horse battery", now).unwrap();
        store.upsert_catalyst(&alice.id, "office", &["vahera".into()], now).unwrap();
        let exp = store.create_experiment(&alice.id, "shared-run", now).unwrap();
        store.grant_experiment(&exp.id, &bob.id, &["vahera".into()], now).unwrap();

        let state = Arc::new(AppState::new(store, signer()).with_profiles_token(Some("secret-1".into())));
        let (status, body) = get(&state, Some("secret-1")).await;
        assert_eq!(status, StatusCode::OK);

        let profiles = body["profiles"].as_array().unwrap();
        assert_eq!(profiles.len(), 2);

        let alice_view = profiles.iter().find(|p| p["email"] == "alice@example.org").unwrap();
        assert_eq!(alice_view["catalysts"][0]["name"], "office");
        assert_eq!(alice_view["experiments_owned"][0]["name"], "shared-run");
        assert_eq!(alice_view["experiments_owned"][0]["grantee_count"], 1);
        assert_eq!(alice_view["experiments_granted"].as_array().unwrap().len(), 0);

        let bob_view = profiles.iter().find(|p| p["email"] == "bob@example.org").unwrap();
        assert_eq!(bob_view["catalysts"].as_array().unwrap().len(), 0);
        assert_eq!(bob_view["experiments_owned"].as_array().unwrap().len(), 0);
        assert_eq!(bob_view["experiments_granted"][0]["id"], exp.id);
        assert_eq!(bob_view["experiments_granted"][0]["capabilities"][0], "vahera");
    }
}
