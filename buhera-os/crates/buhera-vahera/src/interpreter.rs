//! vaHera interpreter.
//!
//! Walks a `Vec<Stmt>` and dispatches each statement against a [`Kernel`].

use std::collections::BTreeMap;

use buhera_kernel::{Kernel, KernelError, MemoryObject, Process, RetrievedItem};
use buhera_substrate::{embed_molecule, embed_text, MoleculeProperties, SCoord};
use thiserror::Error;

use crate::ast::{Stmt, StmtKind};
use crate::parser::{parse_vahera, ParseError};

/// Per-molecule properties keyed by name.
pub type MoleculeDatabase = BTreeMap<String, MoleculeProperties>;

/// A pluggable text-to-coordinate embedder used by the interpreter.
///
/// The vahera crate is intentionally not coupled to any specific
/// embedder implementation. The lexical fallback that ships in
/// `buhera-substrate` is wrapped by [`DefaultEmbedder`]; callers that
/// want semantic embedding pass an implementation from `buhera-embed`.
pub trait Embedder: Send + Sync {
    /// Embed a piece of text into S-entropy space.
    fn embed(&self, text: &str) -> SCoord;
}

/// Default embedder: defers to [`buhera_substrate::embed_text`].
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultEmbedder;

impl Embedder for DefaultEmbedder {
    fn embed(&self, text: &str) -> SCoord {
        embed_text(text)
    }
}

/// Anything a statement may produce.
#[derive(Debug, Clone)]
pub enum NamedResult {
    /// `memory find nearest "..."` hits, paired with the originating
    /// query string so downstream re-rankers can inspect it.
    FindHits {
        /// The query that produced these hits.
        query: String,
        /// The hits, in S-distance order from the kernel.
        hits: Vec<RetrievedItem<MemoryObject>>,
    },
    /// `demon sort` outputs.
    SortedObjects(Vec<MemoryObject>),
    /// `memory list` snapshot.
    ObjectList(Vec<MemoryObject>),
    /// `memory dump <name>` payload.
    Dump {
        /// The name argument.
        name: String,
        /// Matched object, if any.
        obj: Option<MemoryObject>,
    },
    /// `kernel stats` JSON.
    Stats(serde_json::Value),
    /// `kernel trace` log lines.
    Trace(Vec<String>),
    /// `process list` snapshot.
    Processes(Vec<Process>),
}

/// A resolved target (`describe` or `resolve` outcome).
#[derive(Debug, Clone)]
pub struct ResolvedTarget {
    /// Target name.
    pub name: String,
    /// Categorical coordinate.
    pub coord: SCoord,
}

/// Execution context carrying transient state across statements.
#[derive(Debug, Default)]
pub struct ExecContext {
    /// Map from target name to coordinate (populated by `describe`/`resolve`).
    pub targets: BTreeMap<String, SCoord>,
    /// Map from program name to spawned process pid.
    pub processes: BTreeMap<String, u64>,
    /// Per-statement results in execution order.
    pub results: Vec<NamedResult>,
    /// Human-readable interpreter trace.
    pub trace: Vec<String>,
    /// The most recent `memory find nearest` query string, if any.
    /// Used by the REPL/demo for token-overlap re-ranking.
    pub last_query: Option<String>,
}

/// Top-level interpreter error.
#[derive(Debug, Clone, Error)]
pub enum ExecError {
    /// Parse error.
    #[error(transparent)]
    Parse(#[from] ParseError),
    /// Kernel error.
    #[error(transparent)]
    Kernel(#[from] KernelError),
    /// Runtime error from the interpreter itself.
    #[error("runtime: {0}")]
    Runtime(String),
}

/// Parse `source`, then execute on `kernel` using the default
/// (lexical) embedder.
///
/// This is a thin wrapper around [`execute_vahera_with`] that supplies
/// [`DefaultEmbedder`]. Callers that want semantic embedding should
/// use [`execute_vahera_with`] directly with a
/// `buhera_embed::SemanticEmbedder`.
pub fn execute_vahera(
    source: &str,
    kernel: &mut Kernel,
    molecules: &MoleculeDatabase,
) -> Result<ExecContext, ExecError> {
    execute_vahera_with(source, kernel, molecules, &DefaultEmbedder)
}

/// Parse `source`, then execute on `kernel` using `embedder` for all
/// text → coordinate conversions.
pub fn execute_vahera_with(
    source: &str,
    kernel: &mut Kernel,
    molecules: &MoleculeDatabase,
    embedder: &dyn Embedder,
) -> Result<ExecContext, ExecError> {
    let stmts = parse_vahera(source)?;
    let mut ctx = ExecContext::default();
    for stmt in stmts {
        run_one(&stmt, kernel, &mut ctx, molecules, embedder)?;
    }
    Ok(ctx)
}

fn run_one(
    stmt: &Stmt,
    kernel: &mut Kernel,
    ctx: &mut ExecContext,
    molecules: &MoleculeDatabase,
    embedder: &dyn Embedder,
) -> Result<(), ExecError> {
    match &stmt.kind {
        StmtKind::Describe { name, text } => {
            let coord = resolve_coord(name, text, molecules, embedder);
            ctx.trace.push(format!(
                "describe {} -> S({:.3},{:.3},{:.3})",
                name, coord.k, coord.t, coord.e
            ));
            ctx.targets.insert(name.clone(), coord);
        }

        StmtKind::Resolve { name } => {
            if !ctx.targets.contains_key(name) {
                let coord = resolve_coord(name, name, molecules, embedder);
                ctx.targets.insert(name.clone(), coord);
            }
            let coord = ctx.targets[name];
            ctx.trace.push(format!("resolve {} -> {}", name, coord));
        }

        StmtKind::Spawn { program, target } => {
            let s_final = *ctx.targets.get(target).ok_or_else(|| {
                ExecError::Runtime(format!("spawn: unresolved target {}", target))
            })?;
            let s_initial = SCoord::root();
            let p = kernel.spawn(program, s_initial, s_final)?;
            ctx.processes.insert(program.clone(), p.pid);
            ctx.trace.push(format!(
                "spawn {} -> pid={} d_traj={:.3}",
                program,
                p.pid,
                buhera_substrate::s_distance(s_initial, s_final)
            ));
        }

        StmtKind::NavigatePenultimate => {
            let pid = first_pid(ctx).ok_or_else(|| {
                ExecError::Runtime("navigate: no active process".to_string())
            })?;
            let traj = kernel.navigate(pid)?;
            ctx.trace
                .push(format!("navigate pid={} steps={}", pid, traj.steps));
        }

        StmtKind::CompleteTrajectory => {
            let pid = first_pid(ctx).ok_or_else(|| {
                ExecError::Runtime("complete: no active process".to_string())
            })?;
            let coord = kernel.complete(pid)?;
            ctx.trace
                .push(format!("complete pid={} final={}", pid, coord));
        }

        StmtKind::MemoryCreate { coord } => {
            let obj = kernel.allocate(*coord, serde_json::Value::Null, BTreeMap::new())?;
            ctx.trace
                .push(format!("memory_create addr={}", obj.address));
        }

        StmtKind::MemoryStore { name, text } => {
            let coord = embedder.embed(text);
            let mut meta = BTreeMap::new();
            meta.insert("name".to_string(), serde_json::json!(name));
            // Store the original text as part of the payload so the
            // overlap re-ranker can read it back.
            meta.insert("source".to_string(), serde_json::json!(text));
            let obj = kernel.store(coord, serde_json::json!(text), meta)?;
            ctx.trace
                .push(format!("memory_store name={} addr={}", name, obj.address));
        }

        StmtKind::MemoryFind { query, k } => {
            let q_coord = embedder.embed(query);
            let hits = kernel.find_nearest(q_coord, *k);
            ctx.trace.push(format!(
                "memory_find query={:?} -> {} hits",
                query,
                hits.len()
            ));
            ctx.last_query = Some(query.clone());
            ctx.results.push(NamedResult::FindHits {
                query: query.clone(),
                hits,
            });
        }

        StmtKind::MemoryList => {
            let all = kernel.cmm.all_objects();
            ctx.trace.push(format!("memory_list {} objects", all.len()));
            ctx.results.push(NamedResult::ObjectList(all));
        }

        StmtKind::MemoryDump { name } => {
            let matched = kernel
                .cmm
                .all_objects()
                .into_iter()
                .find(|obj| {
                    obj.metadata
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(|s| s == name)
                        .unwrap_or(false)
                });
            ctx.trace.push(format!("memory_dump name={}", name));
            ctx.results.push(NamedResult::Dump {
                name: name.clone(),
                obj: matched,
            });
        }

        StmtKind::DemonSort => {
            let items: Vec<(SCoord, MemoryObject)> = kernel
                .cmm
                .all_objects()
                .into_iter()
                .map(|o| (o.coord, o))
                .collect();
            let sorted = kernel.dic.categorical_sort(&items);
            let objs: Vec<MemoryObject> = sorted.into_iter().map(|(_, o)| o).collect();
            ctx.trace.push(format!("demon_sort {} items", objs.len()));
            ctx.results.push(NamedResult::SortedObjects(objs));
        }

        StmtKind::ControllerVerify => {
            let stats = kernel.tem.stats();
            ctx.trace.push(format!(
                "controller_verify samples={} alerts={} max_delta={:.5}",
                stats.samples, stats.alerts, stats.max_delta
            ));
        }

        StmtKind::KernelStats => {
            let stats = kernel.stats();
            ctx.trace.push(format!(
                "kernel_stats cmm_objects={} verified={} samples={}",
                stats.cmm_objects, stats.pve.verified, stats.tem.samples
            ));
            ctx.results
                .push(NamedResult::Stats(serde_json::to_value(stats).unwrap_or_default()));
        }

        StmtKind::KernelTrace => {
            let log = kernel.activity_log();
            ctx.trace.push(format!("kernel_trace {} entries", log.len()));
            ctx.results.push(NamedResult::Trace(log));
        }

        StmtKind::ProcessList => {
            let procs = kernel.pss.all();
            ctx.trace.push(format!("process_list {} processes", procs.len()));
            ctx.results.push(NamedResult::Processes(procs));
        }
    }

    Ok(())
}

fn first_pid(ctx: &ExecContext) -> Option<u64> {
    ctx.processes.values().next().copied()
}

/// Decide which embedding to use for a `describe`/`resolve` target.
///
/// If the name (case-insensitively) matches a known molecule, use
/// [`embed_molecule`]. Otherwise defer to the supplied `embedder`.
fn resolve_coord(
    name: &str,
    text: &str,
    molecules: &MoleculeDatabase,
    embedder: &dyn Embedder,
) -> SCoord {
    if molecules.is_empty() {
        return embedder.embed(text);
    }
    let name_lc = name.to_lowercase();
    for (mol_name, props) in molecules {
        if mol_name.to_lowercase() == name_lc {
            return embed_molecule(mol_name, props);
        }
    }
    // Token match in name or text.
    let mut tokens: Vec<String> = name_lc
        .split(|c: char| c.is_whitespace() || c == '_' || c == '-')
        .map(String::from)
        .collect();
    tokens.extend(text.to_lowercase().split_whitespace().map(String::from));
    for (mol_name, props) in molecules {
        let mn = mol_name.to_lowercase();
        if tokens.iter().any(|t| t == &mn) {
            return embed_molecule(mol_name, props);
        }
    }
    embedder.embed(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use buhera_kernel::Kernel;

    #[test]
    fn round_trip_describe_spawn_navigate_complete() {
        let src = r#"
describe ethanol with "ethanol C2H5OH bp 78"
spawn lookup_eth from ethanol
navigate to penultimate
complete trajectory
"#;
        let mut k = Kernel::with_default_depth();
        let molecules = MoleculeDatabase::new();
        let ctx = execute_vahera(src, &mut k, &molecules).unwrap();
        assert!(ctx.targets.contains_key("ethanol"));
        assert!(ctx.processes.contains_key("lookup_eth"));
    }

    #[test]
    fn memory_store_and_find() {
        let src = r#"
memory store "greeting" = "hello world"
memory store "question" = "what is the boiling point of ethanol?"
memory find nearest "boiling point" k=1
"#;
        let mut k = Kernel::with_default_depth();
        let molecules = MoleculeDatabase::new();
        let ctx = execute_vahera(src, &mut k, &molecules).unwrap();
        assert_eq!(k.cmm.len(), 2);
        let hits = ctx.results.iter().find_map(|r| match r {
            NamedResult::FindHits { hits, .. } => Some(hits.clone()),
            _ => None,
        });
        assert!(hits.unwrap().len() >= 1);
    }
}
