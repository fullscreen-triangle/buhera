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
//! * `/api/run`         — submit work; the router decides where it executes.
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
}

impl AppState {
    /// Build the shared state.
    pub fn new(store: Store, signer: Signer) -> Self {
        Self {
            store: Mutex::new(store),
            signer,
            sessions: session::Sessions::new(),
        }
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
            // The relay lands next. Until a machine can actually be reached,
            // say so plainly rather than silently running it here — that
            // would return an answer computed against the wrong filesystem.
            Err(ApiError::Unroutable(format!(
                "machine {name:?} is live but the relay is not yet implemented; \
                 retry with prefer=\"gateway\" to run on the server"
            )))
        }
    }
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
        .route("/api/catalysts/:name", axum::routing::delete(unpair_catalyst))
        .route("/api/run", post(run))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .with_state(state)
}
