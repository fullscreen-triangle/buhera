//! Buhera OS binary support library.
//!
//! Shared between the `demo` and `repl` binaries: boots a kernel with a
//! NIST compound database, loads molecule properties, renders results.

use std::collections::BTreeMap;
use std::path::Path;

use buhera_kernel::{Kernel, MemoryObject, RetrievedItem};
use buhera_substrate::{embed_molecule, MoleculeProperties};
use buhera_vahera::{MoleculeDatabase, NamedResult};

/// Load a JSON compound database from disk.
///
/// File format: `{ "name": { "formula": "...", "molecular_weight": <f64>,
/// "boiling_point_c": <f64>, "n_atoms": <f64>, ... }, ... }`.
pub fn load_nist(path: impl AsRef<Path>) -> std::io::Result<NistDatabase> {
    let text = std::fs::read_to_string(path)?;
    let raw: BTreeMap<String, BTreeMap<String, serde_json::Value>> =
        serde_json::from_str(&text).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let mut molecules = MoleculeDatabase::new();
    for (name, props) in &raw {
        molecules.insert(name.clone(), MoleculeProperties::from_map(props));
    }

    Ok(NistDatabase { molecules, raw })
}

/// Compound database + raw payloads for kernel allocation.
#[derive(Debug, Clone)]
pub struct NistDatabase {
    /// Properties indexed by name, for embedding.
    pub molecules: MoleculeDatabase,
    /// Raw JSON properties for opaque storage.
    pub raw: BTreeMap<String, BTreeMap<String, serde_json::Value>>,
}

/// Boot a kernel and allocate every compound in `db` at its categorical
/// coordinate.
pub fn boot_os(db: &NistDatabase, depth: usize) -> Kernel {
    let mut k = Kernel::new(depth);
    for (name, props) in &db.molecules {
        let coord = embed_molecule(name, props);
        let payload = serde_json::Value::Object(db.raw[name].clone().into_iter().collect());
        let mut meta = BTreeMap::new();
        meta.insert("name".to_string(), serde_json::json!(name));
        if let Some(f) = db.raw[name].get("formula") {
            meta.insert("formula".to_string(), f.clone());
        }
        let _ = k.allocate(coord, payload, meta);
    }
    k
}

/// Render a list of `RetrievedItem<MemoryObject>` in REPL-friendly form.
pub fn render_hits(hits: &[RetrievedItem<MemoryObject>]) -> String {
    let mut s = String::new();
    if hits.is_empty() {
        s.push_str("  (no hits)\n");
        return s;
    }
    for (i, h) in hits.iter().enumerate() {
        let name = h
            .value
            .metadata
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        s.push_str(&format!(
            "  [{}] name={} addr={} d={:.4}\n",
            i + 1,
            name,
            &h.value.address[..h.value.address.len().min(12)],
            h.distance
        ));
    }
    s
}

/// Render a list of memory objects.
pub fn render_objects(objs: &[MemoryObject]) -> String {
    let mut s = String::new();
    for obj in objs {
        let name = obj
            .metadata
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        s.push_str(&format!(
            "  {} {} tier={} coord={}\n",
            &obj.address[..obj.address.len().min(12)],
            name,
            obj.tier.as_str(),
            obj.coord
        ));
    }
    s
}

/// Pretty-print one `NamedResult` to stdout.
pub fn print_result(result: &NamedResult) {
    match result {
        NamedResult::FindHits(hits) => {
            println!("hits:");
            print!("{}", render_hits(hits));
        }
        NamedResult::SortedObjects(objs) => {
            println!("sorted ({} objects):", objs.len());
            print!("{}", render_objects(objs));
        }
        NamedResult::ObjectList(objs) => {
            println!("memory ({} objects):", objs.len());
            print!("{}", render_objects(objs));
        }
        NamedResult::Dump { name, obj } => match obj {
            Some(o) => {
                println!("dump {}:", name);
                println!("  address: {}", o.address);
                println!("  coord:   {}", o.coord);
                println!("  tier:    {}", o.tier.as_str());
                println!(
                    "  payload: {}",
                    serde_json::to_string_pretty(&o.payload).unwrap_or_default()
                );
            }
            None => println!("dump {}: (not found)", name),
        },
        NamedResult::Stats(stats) => {
            println!(
                "stats: {}",
                serde_json::to_string_pretty(stats).unwrap_or_default()
            );
        }
        NamedResult::Trace(log) => {
            println!("trace ({} entries):", log.len());
            for line in log {
                println!("  {}", line);
            }
        }
        NamedResult::Processes(procs) => {
            println!("processes ({}):", procs.len());
            for p in procs {
                println!(
                    "  pid={} name={} state={} d_traj={:.3}",
                    p.pid,
                    p.program_name,
                    p.state.as_str(),
                    buhera_substrate::s_distance(p.s_current, p.s_final)
                );
            }
        }
    }
}
