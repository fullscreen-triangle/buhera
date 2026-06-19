//! Buhera OS HTTP server.
//!
//! Wraps a single `Kernel` + `Box<dyn TextEmbedder>` behind a small
//! JSON API so browser clients (or any HTTP client) can talk to a live
//! kernel without bundling Rust into the browser.
//!
//! The server keeps one kernel instance for the lifetime of the
//! process; all requests share it. Use `:clear` / `DELETE /memory` to
//! reset.

use std::net::SocketAddr;
use std::process::ExitCode;
use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::{HeaderValue, Method, StatusCode},
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use buhera_embed::{LexicalEmbedder, SemanticEmbedder, TextEmbedder};
use buhera_kernel::Kernel;
use buhera_vahera::{execute_vahera_with, Embedder as VaheraEmbedder, MoleculeDatabase, NamedResult};
use buhera_substrate::SCoord;
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

// ─────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug, Clone)]
#[command(
    name = "buhera-server",
    about = "HTTP/JSON server fronting a live Buhera kernel."
)]
struct Args {
    /// Bind address (default 127.0.0.1:5599).
    #[arg(long, default_value = "127.0.0.1:5599")]
    bind: String,

    /// Ternary-address depth used by CMM.
    #[arg(long, default_value_t = 12)]
    depth: usize,

    /// Use the lexical hash-bag embedder instead of the semantic model.
    #[arg(long, default_value_t = false)]
    lexical: bool,

    /// Allow CORS from any origin. Default true so the local webtool
    /// at http://localhost:3000 can call the API.
    #[arg(long, default_value_t = true)]
    permissive_cors: bool,
}

// ─────────────────────────────────────────────────────────────────────
// Shared state
// ─────────────────────────────────────────────────────────────────────

struct AppState {
    kernel: Mutex<Kernel>,
    embedder: Box<dyn TextEmbedder>,
    embedder_name: String,
    depth: usize,
    molecules: MoleculeDatabase,
}

/// Bridge between `buhera-embed::TextEmbedder` and the
/// `buhera-vahera::Embedder` trait expected by the interpreter.
struct EmbedderAdapter<'a> {
    inner: &'a dyn TextEmbedder,
}

impl<'a> VaheraEmbedder for EmbedderAdapter<'a> {
    fn embed(&self, text: &str) -> SCoord {
        self.inner.embed(text)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Request / response types
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct VaheraRequest {
    source: String,
    /// If true, apply the token-overlap re-ranker to `FindHits` results.
    #[serde(default = "default_overlap")]
    rerank: bool,
}

fn default_overlap() -> bool {
    true
}

#[derive(Debug, Serialize)]
struct VaheraResponse {
    trace: Vec<String>,
    results: Vec<NamedResultDto>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum NamedResultDto {
    FindHits {
        query: String,
        hits: Vec<HitDto>,
    },
    ObjectList {
        objects: Vec<ObjectDto>,
    },
    SortedObjects {
        objects: Vec<ObjectDto>,
    },
    Dump {
        name: String,
        object: Option<ObjectDto>,
    },
    Stats {
        stats: serde_json::Value,
    },
    Trace {
        log: Vec<String>,
    },
    Processes {
        processes: Vec<ProcessDto>,
    },
}

#[derive(Debug, Serialize)]
struct HitDto {
    name: Option<String>,
    address: String,
    coord: [f64; 3],
    tier: String,
    distance: f64,
    source: Option<String>,
}

#[derive(Debug, Serialize)]
struct ObjectDto {
    name: Option<String>,
    address: String,
    coord: [f64; 3],
    tier: String,
    source: Option<String>,
}

#[derive(Debug, Serialize)]
struct ProcessDto {
    pid: u64,
    program_name: String,
    state: String,
    s_current: [f64; 3],
    s_final: [f64; 3],
    d_traj: f64,
}

#[derive(Debug, Serialize)]
struct InfoResponse {
    version: &'static str,
    embedder: String,
    depth: usize,
    objects: usize,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

// ─────────────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────────────

async fn info(State(s): State<Arc<AppState>>) -> Json<InfoResponse> {
    let kernel = s.kernel.lock().await;
    Json(InfoResponse {
        version: env!("CARGO_PKG_VERSION"),
        embedder: s.embedder_name.clone(),
        depth: s.depth,
        objects: kernel.cmm.len(),
    })
}

async fn run_vahera(
    State(s): State<Arc<AppState>>,
    Json(req): Json<VaheraRequest>,
) -> Result<Json<VaheraResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut kernel = s.kernel.lock().await;
    let adapter = EmbedderAdapter {
        inner: s.embedder.as_ref(),
    };

    let mut ctx = match execute_vahera_with(&req.source, &mut kernel, &s.molecules, &adapter) {
        Ok(c) => c,
        Err(err) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: err.to_string(),
                }),
            ));
        }
    };

    if req.rerank {
        for r in &mut ctx.results {
            if let NamedResult::FindHits { query, hits } = r {
                buhera_os::rerank_hits_with_overlap(query, hits, 0.5);
            }
        }
    }

    Ok(Json(VaheraResponse {
        trace: ctx.trace,
        results: ctx.results.into_iter().map(to_dto).collect(),
    }))
}

async fn list_objects(State(s): State<Arc<AppState>>) -> Json<Vec<ObjectDto>> {
    let kernel = s.kernel.lock().await;
    let objs = kernel.cmm.all_objects().into_iter().map(obj_to_dto).collect();
    Json(objs)
}

#[derive(Deserialize)]
struct FindParams {
    q: String,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default = "default_overlap")]
    rerank: bool,
}

fn default_k() -> usize {
    5
}

async fn find(
    State(s): State<Arc<AppState>>,
    Query(p): Query<FindParams>,
) -> Json<Vec<HitDto>> {
    let mut kernel = s.kernel.lock().await;
    let q_coord = s.embedder.embed(&p.q);
    let mut hits = kernel.find_nearest(q_coord, p.k);
    if p.rerank {
        buhera_os::rerank_hits_with_overlap(&p.q, &mut hits, 0.5);
    }
    Json(hits.into_iter().map(hit_to_dto).collect())
}

#[derive(Deserialize)]
struct StoreReq {
    name: String,
    text: String,
}

async fn store(
    State(s): State<Arc<AppState>>,
    Json(req): Json<StoreReq>,
) -> Result<Json<ObjectDto>, (StatusCode, Json<ErrorResponse>)> {
    let coord = s.embedder.embed(&req.text);
    let mut kernel = s.kernel.lock().await;
    let mut meta = std::collections::BTreeMap::new();
    meta.insert("name".to_string(), serde_json::json!(req.name));
    meta.insert("source".to_string(), serde_json::json!(req.text));
    match kernel.store(coord, serde_json::json!(req.text), meta) {
        Ok(obj) => Ok(Json(obj_to_dto(obj))),
        Err(err) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )),
    }
}

async fn clear(State(s): State<Arc<AppState>>) -> Json<InfoResponse> {
    let mut kernel = s.kernel.lock().await;
    *kernel = Kernel::new(s.depth);
    Json(InfoResponse {
        version: env!("CARGO_PKG_VERSION"),
        embedder: s.embedder_name.clone(),
        depth: s.depth,
        objects: kernel.cmm.len(),
    })
}

async fn stats(State(s): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let kernel = s.kernel.lock().await;
    Json(serde_json::to_value(kernel.stats()).unwrap_or_default())
}

async fn health() -> impl IntoResponse {
    "ok"
}

// ─────────────────────────────────────────────────────────────────────
// DTO conversion
// ─────────────────────────────────────────────────────────────────────

fn to_dto(r: NamedResult) -> NamedResultDto {
    match r {
        NamedResult::FindHits { query, hits } => NamedResultDto::FindHits {
            query,
            hits: hits.into_iter().map(hit_to_dto).collect(),
        },
        NamedResult::ObjectList(objs) => NamedResultDto::ObjectList {
            objects: objs.into_iter().map(obj_to_dto).collect(),
        },
        NamedResult::SortedObjects(objs) => NamedResultDto::SortedObjects {
            objects: objs.into_iter().map(obj_to_dto).collect(),
        },
        NamedResult::Dump { name, obj } => NamedResultDto::Dump {
            name,
            object: obj.map(obj_to_dto),
        },
        NamedResult::Stats(s) => NamedResultDto::Stats { stats: s },
        NamedResult::Trace(log) => NamedResultDto::Trace { log },
        NamedResult::Processes(procs) => NamedResultDto::Processes {
            processes: procs
                .into_iter()
                .map(|p| ProcessDto {
                    pid: p.pid,
                    program_name: p.program_name,
                    state: p.state.as_str().to_string(),
                    s_current: [p.s_current.k, p.s_current.t, p.s_current.e],
                    s_final: [p.s_final.k, p.s_final.t, p.s_final.e],
                    d_traj: buhera_substrate::s_distance(p.s_current, p.s_final),
                })
                .collect(),
        },
    }
}

fn hit_to_dto(h: buhera_kernel::RetrievedItem<buhera_kernel::MemoryObject>) -> HitDto {
    HitDto {
        name: h
            .value
            .metadata
            .get("name")
            .and_then(|v| v.as_str())
            .map(String::from),
        address: h.value.address.clone(),
        coord: [h.value.coord.k, h.value.coord.t, h.value.coord.e],
        tier: h.value.tier.as_str().to_string(),
        distance: h.distance,
        source: h
            .value
            .metadata
            .get("source")
            .and_then(|v| v.as_str())
            .map(String::from),
    }
}

fn obj_to_dto(obj: buhera_kernel::MemoryObject) -> ObjectDto {
    ObjectDto {
        name: obj
            .metadata
            .get("name")
            .and_then(|v| v.as_str())
            .map(String::from),
        address: obj.address.clone(),
        coord: [obj.coord.k, obj.coord.t, obj.coord.e],
        tier: obj.tier.as_str().to_string(),
        source: obj
            .metadata
            .get("source")
            .and_then(|v| v.as_str())
            .map(String::from),
    }
}

// ─────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let (embedder, embedder_name): (Box<dyn TextEmbedder>, String) = if args.lexical {
        let e = LexicalEmbedder::new();
        let n = e.name().to_string();
        (Box::new(e), n)
    } else {
        eprintln!("(loading semantic embedder; first run downloads ~33 MB)");
        match SemanticEmbedder::new() {
            Ok(e) => {
                let n = e.name().to_string();
                (Box::new(e), n)
            }
            Err(err) => {
                eprintln!(
                    "(semantic embedder failed: {}; falling back to lexical)",
                    err
                );
                let e = LexicalEmbedder::new();
                let n = e.name().to_string();
                (Box::new(e), n)
            }
        }
    };

    let state = Arc::new(AppState {
        kernel: Mutex::new(Kernel::new(args.depth)),
        embedder,
        embedder_name: embedder_name.clone(),
        depth: args.depth,
        molecules: MoleculeDatabase::new(),
    });

    let cors = if args.permissive_cors {
        CorsLayer::new()
            .allow_methods([Method::GET, Method::POST, Method::DELETE])
            .allow_origin(tower_http::cors::Any)
            .allow_headers([axum::http::header::CONTENT_TYPE])
    } else {
        CorsLayer::new()
            .allow_methods([Method::GET, Method::POST, Method::DELETE])
            .allow_origin(
                "http://localhost:3000"
                    .parse::<HeaderValue>()
                    .expect("static origin"),
            )
            .allow_headers([axum::http::header::CONTENT_TYPE])
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/info", get(info))
        .route("/stats", get(stats))
        .route("/vahera", post(run_vahera))
        .route("/store", post(store))
        .route("/find", get(find))
        .route("/list", get(list_objects))
        .route("/memory", delete(clear))
        .with_state(state.clone())
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = match args.bind.parse() {
        Ok(a) => a,
        Err(err) => {
            eprintln!("invalid --bind {}: {}", args.bind, err);
            return ExitCode::from(2);
        }
    };

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(err) => {
            eprintln!("could not bind {}: {}", addr, err);
            return ExitCode::from(2);
        }
    };

    println!(
        "buhera-server  v{}  bound {}  embedder={}  depth={}",
        env!("CARGO_PKG_VERSION"),
        addr,
        embedder_name,
        args.depth
    );
    println!("endpoints:");
    println!("  GET  /info");
    println!("  GET  /stats");
    println!("  GET  /list");
    println!("  GET  /find?q=...&k=N&rerank=true");
    println!("  POST /store           {{\"name\":..., \"text\":...}}");
    println!("  POST /vahera          {{\"source\":..., \"rerank\":bool}}");
    println!("  DELETE /memory        clear the kernel");
    println!();

    if let Err(err) = axum::serve(listener, app).await {
        eprintln!("server error: {}", err);
        return ExitCode::from(1);
    }

    ExitCode::SUCCESS
}
