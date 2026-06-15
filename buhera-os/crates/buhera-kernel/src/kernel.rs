//! Kernel orchestrator.
//!
//! Threads requests through the five subsystems in the canonical order:
//! PVE → CMM/PSS/DIC (the doer) → TEM (after the fact).

use std::collections::BTreeMap;

use buhera_substrate::{backward_navigate, completion_morphism, s_distance, SCoord, Trajectory};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::cmm::{Cmm, MemoryObject};
use crate::dic::{Dic, DicStats, RetrievedItem};
use crate::pss::{Process, Pss};
use crate::pve::{Pve, PveError, PveStats};
use crate::tem::{Tem, TemStats};

/// Top-level kernel error.
#[derive(Debug, Clone, Error)]
pub enum KernelError {
    /// Pre-dispatch validation failed.
    #[error("PVE rejected: {0}")]
    Pve(#[from] PveError),
    /// The requested process is not known to the scheduler.
    #[error("unknown pid: {0}")]
    UnknownPid(u64),
    /// Generic runtime error from the kernel.
    #[error("kernel: {0}")]
    Runtime(String),
}

/// Snapshot of per-subsystem statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelStats {
    /// Number of objects in CMM.
    pub cmm_objects: usize,
    /// DIC stats.
    pub dic: DicStats,
    /// PVE stats.
    pub pve: PveStats,
    /// TEM stats.
    pub tem: TemStats,
}

/// The Buhera kernel: a single struct holding all five subsystems plus
/// a unified trace.
#[derive(Debug)]
pub struct Kernel {
    /// Categorical Memory Manager.
    pub cmm: Cmm,
    /// Penultimate State Scheduler.
    pub pss: Pss,
    /// Demon I/O Controller.
    pub dic: Dic,
    /// Proof Validation Engine.
    pub pve: Pve,
    /// Triple Equivalence Monitor.
    pub tem: Tem,
    trace: Vec<String>,
}

impl Kernel {
    /// Construct a kernel with the given ternary-address depth.
    pub fn new(depth: usize) -> Self {
        Self {
            cmm: Cmm::new(depth),
            pss: Pss::new(),
            dic: Dic::new(),
            pve: Pve::new(),
            tem: Tem::with_default_threshold(),
            trace: Vec::new(),
        }
    }

    /// Construct a kernel with a default depth of 12.
    pub fn with_default_depth() -> Self {
        Self::new(12)
    }

    // ─────────── interpreter-facing primitives ───────────

    /// Allocate a memory object at `coord`.
    pub fn allocate(
        &mut self,
        coord: SCoord,
        payload: serde_json::Value,
        metadata: BTreeMap<String, serde_json::Value>,
    ) -> Result<MemoryObject, KernelError> {
        let mut payload_map = BTreeMap::new();
        payload_map.insert(
            "coord".to_string(),
            serde_json::json!({"k": coord.k, "t": coord.t, "e": coord.e}),
        );
        self.pve.verify("memory_create", &payload_map)?;
        let obj = self.cmm.allocate(coord, payload, metadata);
        self.tem
            .sample(coord, &format!("allocate addr={}", obj.address));
        self.trace
            .push(format!("ALLOCATE coord={} -> addr={}", coord, obj.address));
        Ok(obj)
    }

    /// Convenience: store `data` at `coord` with optional metadata.
    pub fn store(
        &mut self,
        coord: SCoord,
        data: serde_json::Value,
        metadata: BTreeMap<String, serde_json::Value>,
    ) -> Result<MemoryObject, KernelError> {
        self.allocate(coord, data, metadata)
    }

    /// Find the `k` memory objects nearest to `target`.
    pub fn find_nearest(
        &mut self,
        target: SCoord,
        k: usize,
    ) -> Vec<RetrievedItem<MemoryObject>> {
        // CMM proximity scan (oversample to give DIC something to prune).
        let candidates = self.cmm.proximity_query(target, k * 3);
        let source: Vec<(SCoord, MemoryObject)> = candidates
            .into_iter()
            .map(|(obj, _)| (obj.coord, obj))
            .collect();
        let retrieved = self.dic.retrieve(&source, target, k);
        self.trace
            .push(format!("FIND_NEAREST target={} -> {} results", target, retrieved.len()));
        retrieved
    }

    /// Spawn a new categorical process.
    pub fn spawn(
        &mut self,
        program_name: &str,
        s_initial: SCoord,
        s_final: SCoord,
    ) -> Result<Process, KernelError> {
        let mut payload = BTreeMap::new();
        payload.insert("target".to_string(), serde_json::json!(program_name));
        self.pve.verify("resolve", &payload)?;
        let p = self.pss.spawn(program_name, s_initial, s_final);
        self.tem.sample(s_final, &format!("spawn pid={}", p.pid));
        self.trace.push(format!(
            "SPAWN pid={} {} d_traj={:.3}",
            p.pid,
            program_name,
            s_distance(s_initial, s_final)
        ));
        Ok(p)
    }

    /// Backward-navigate `pid` to the penultimate state of its target.
    pub fn navigate(&mut self, pid: u64) -> Result<Trajectory, KernelError> {
        let proc = self
            .pss
            .get(pid)
            .ok_or(KernelError::UnknownPid(pid))?;
        let s_final = proc.s_final;

        let mut payload = BTreeMap::new();
        payload.insert("mode".to_string(), serde_json::json!("penultimate"));
        self.pve.verify("navigate", &payload)?;

        let traj = backward_navigate(s_final, self.cmm.depth);
        let penultimate = if traj.path.len() >= 2 {
            traj.path[traj.path.len() - 2]
        } else {
            traj.path[0]
        };
        self.pss.advance(pid, penultimate);
        self.tem
            .sample(penultimate, &format!("navigate pid={}", pid));
        self.trace.push(format!(
            "NAVIGATE pid={} {} steps miracles={}",
            pid, traj.steps, traj.miracle_count
        ));
        Ok(traj)
    }

    /// Apply the completion morphism from penultimate to final.
    pub fn complete(&mut self, pid: u64) -> Result<SCoord, KernelError> {
        let proc = self
            .pss
            .get(pid)
            .ok_or(KernelError::UnknownPid(pid))?;
        let penultimate = proc.s_current;
        let s_final = proc.s_final;

        let mut payload = BTreeMap::new();
        payload.insert(
            "s_penultimate".to_string(),
            serde_json::json!({"k": penultimate.k, "t": penultimate.t, "e": penultimate.e}),
        );
        self.pve.verify("complete", &payload)?;

        let new_coord = completion_morphism(penultimate, s_final);
        self.pss.advance(pid, new_coord);
        self.tem.sample(new_coord, &format!("complete pid={}", pid));
        self.trace.push(format!("COMPLETE pid={}", pid));
        Ok(new_coord)
    }

    // ─────────── diagnostics ───────────

    /// Unified activity trace across all subsystems.
    pub fn activity_log(&self) -> Vec<String> {
        let mut log = Vec::new();
        log.extend(self.cmm.events());
        log.extend(self.pss.events());
        log.extend(self.dic.events());
        log.extend(self.pve.events());
        log.extend(self.tem.events());
        log
    }

    /// Kernel-level trace (one entry per orchestrator call).
    pub fn trace(&self) -> Vec<String> {
        self.trace.clone()
    }

    /// Snapshot of all subsystem stats.
    pub fn stats(&self) -> KernelStats {
        KernelStats {
            cmm_objects: self.cmm.len(),
            dic: self.dic.stats(),
            pve: self.pve.stats(),
            tem: self.tem.stats(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn allocate_navigate_complete_round_trip() {
        let mut k = Kernel::with_default_depth();
        let s = SCoord::new(0.4, 0.5, 0.6).unwrap();
        let _ = k.allocate(s, json!({"v": 1}), BTreeMap::new()).unwrap();
        let proc = k.spawn("p", SCoord::root(), s).unwrap();
        let _ = k.navigate(proc.pid).unwrap();
        let final_coord = k.complete(proc.pid).unwrap();
        assert_eq!(final_coord, s);
        assert!(k.stats().cmm_objects == 1);
    }
}
