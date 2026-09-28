//! Wire rendering of interpreter results (moved from `buhera-gateway`).

use buhera_kernel::MemoryObject;

use crate::NamedResult;

/// Render one interpreter result as JSON for the wire.
///
/// Shared by the gateway's `/api/run` path and the `vahera` registry module
/// (`buhera-modules`), so a statement renders identically on both.
pub fn render_result(r: &NamedResult) -> serde_json::Value {
    match r {
        NamedResult::FindHits { query, hits } => serde_json::json!({
            "kind": "hits",
            "query": query,
            "hits": hits.iter().map(|h| serde_json::json!({
                "name": h.value.metadata.get("name").and_then(|v| v.as_str()).unwrap_or("?"),
                "address": h.value.address,
                "distance": h.distance,
            })).collect::<Vec<_>>(),
        }),
        NamedResult::SortedObjects(objs) => serde_json::json!({
            "kind": "sorted",
            "objects": objs.iter().map(brief).collect::<Vec<_>>(),
        }),
        NamedResult::ObjectList(objs) => serde_json::json!({
            "kind": "list",
            "objects": objs.iter().map(brief).collect::<Vec<_>>(),
        }),
        NamedResult::Dump { name, obj } => serde_json::json!({
            "kind": "dump",
            "name": name,
            "object": obj.as_ref().map(|o| serde_json::json!({
                "address": o.address,
                "coord": { "k": o.coord.k, "t": o.coord.t, "e": o.coord.e },
                "tier": o.tier.as_str(),
                "payload": o.payload,
            })),
        }),
        NamedResult::Stats(v) => serde_json::json!({ "kind": "stats", "stats": v }),
        NamedResult::Trace(lines) => serde_json::json!({ "kind": "trace", "lines": lines }),
        NamedResult::Processes(ps) => serde_json::json!({
            "kind": "processes",
            "processes": ps.iter().map(|p| serde_json::json!({
                "pid": p.pid,
                "program": p.program_name,
                "state": p.state.as_str(),
            })).collect::<Vec<_>>(),
        }),
    }
}

fn brief(o: &MemoryObject) -> serde_json::Value {
    serde_json::json!({
        "name": o.metadata.get("name").and_then(|v| v.as_str()).unwrap_or("?"),
        "address": o.address,
        "tier": o.tier.as_str(),
        "coord": { "k": o.coord.k, "t": o.coord.t, "e": o.coord.e },
    })
}
