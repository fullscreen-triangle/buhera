//! Penultimate State Scheduler.
//!
//! Processes are ordered by trajectory distance `d_traj = d(S_cur, S_f)`.
//! The next scheduled process is the one closest to its final state.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, BTreeMap};

use buhera_substrate::{s_distance, SCoord};
use serde::{Deserialize, Serialize};

/// Lifecycle state of a [`Process`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessState {
    /// Waiting in the ready queue.
    Ready,
    /// Currently dispatched.
    Running,
    /// Reached its final state (within 1e-3 of the target).
    Completed,
    /// Blocked by an external signal.
    Blocked,
}

impl ProcessState {
    /// String tag matching the Python convention.
    pub fn as_str(&self) -> &'static str {
        match self {
            ProcessState::Ready => "ready",
            ProcessState::Running => "running",
            ProcessState::Completed => "completed",
            ProcessState::Blocked => "blocked",
        }
    }
}

/// A categorical process.
#[derive(Debug, Clone)]
pub struct Process {
    /// Process id.
    pub pid: u64,
    /// Human-readable program name.
    pub program_name: String,
    /// Where the process was spawned.
    pub s_initial: SCoord,
    /// Target coordinate.
    pub s_final: SCoord,
    /// Latest position.
    pub s_current: SCoord,
    /// Lifecycle state.
    pub state: ProcessState,
    /// Per-process event log.
    pub events: Vec<String>,
}

/// Heap entry. We invert ordering so the smallest distance pops first.
#[derive(Debug, Clone)]
struct Entry {
    distance: f64,
    tiebreaker: u64,
    pid: u64,
}

impl Eq for Entry {}
impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; invert so smaller distance is "larger".
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.tiebreaker.cmp(&self.tiebreaker))
    }
}

/// Penultimate State Scheduler.
#[derive(Debug, Default)]
pub struct Pss {
    heap: BinaryHeap<Entry>,
    processes: BTreeMap<u64, Process>,
    next_pid: u64,
    next_tie: u64,
    events: Vec<String>,
}

impl Pss {
    /// Construct an empty scheduler.
    pub fn new() -> Self {
        Self {
            next_pid: 1,
            ..Default::default()
        }
    }

    /// Spawn a new process.
    pub fn spawn(&mut self, program_name: &str, s_initial: SCoord, s_final: SCoord) -> Process {
        let pid = self.next_pid;
        self.next_pid += 1;
        let tie = self.next_tie;
        self.next_tie += 1;
        let p = Process {
            pid,
            program_name: program_name.to_string(),
            s_initial,
            s_final,
            s_current: s_initial,
            state: ProcessState::Ready,
            events: Vec::new(),
        };
        let d = s_distance(s_initial, s_final);
        self.processes.insert(pid, p.clone());
        self.heap.push(Entry {
            distance: d,
            tiebreaker: tie,
            pid,
        });
        self.events.push(format!(
            "PSS.spawn pid={} program={} d_traj={:.3}",
            pid, program_name, d
        ));
        p
    }

    /// Pop the process with the minimum trajectory distance and transition
    /// it to `Running`. Returns `None` if no ready process is available.
    pub fn next(&mut self) -> Option<Process> {
        while let Some(entry) = self.heap.pop() {
            let p = self.processes.get_mut(&entry.pid)?;
            if p.state == ProcessState::Ready {
                p.state = ProcessState::Running;
                self.events.push(format!(
                    "PSS.schedule pid={} d_traj={:.3}",
                    p.pid, entry.distance
                ));
                return Some(p.clone());
            }
        }
        None
    }

    /// Advance a process to a new coordinate; mark it completed if it
    /// reached its target.
    pub fn advance(&mut self, pid: u64, new_coord: SCoord) {
        let final_coord = match self.processes.get(&pid) {
            Some(p) => p.s_final,
            None => return,
        };
        let d = s_distance(new_coord, final_coord);
        if let Some(p) = self.processes.get_mut(&pid) {
            p.s_current = new_coord;
            p.events
                .push(format!("advance to {} d_traj={:.3}", new_coord, d));
            if d < 1e-3 {
                p.state = ProcessState::Completed;
                self.events.push(format!("PSS.complete pid={}", pid));
            } else {
                p.state = ProcessState::Ready;
                let tie = self.next_tie;
                self.next_tie += 1;
                self.heap.push(Entry {
                    distance: d,
                    tiebreaker: tie,
                    pid,
                });
            }
        }
    }

    /// Look up a process by pid.
    pub fn get(&self, pid: u64) -> Option<&Process> {
        self.processes.get(&pid)
    }

    /// All known processes, including completed ones.
    pub fn all(&self) -> Vec<Process> {
        self.processes.values().cloned().collect()
    }

    /// Activity events.
    pub fn events(&self) -> Vec<String> {
        self.events.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_assigns_unique_pids() {
        let mut p = Pss::new();
        let s0 = SCoord::root();
        let s1 = SCoord::new(0.5, 0.5, 0.5).unwrap();
        let a = p.spawn("a", s0, s1);
        let b = p.spawn("b", s0, s1);
        assert_eq!(a.pid, 1);
        assert_eq!(b.pid, 2);
    }

    #[test]
    fn closest_process_dispatches_first() {
        let mut p = Pss::new();
        let s0 = SCoord::root();
        let _far = p.spawn("far", s0, SCoord::new(0.0, 1.0, 1.0).unwrap());
        let near = p.spawn("near", s0, SCoord::new(0.9, 0.05, 0.05).unwrap());
        let scheduled = p.next().unwrap();
        assert_eq!(scheduled.pid, near.pid);
    }

    #[test]
    fn advance_to_target_completes() {
        let mut p = Pss::new();
        let s0 = SCoord::root();
        let s1 = SCoord::new(0.5, 0.5, 0.5).unwrap();
        let proc = p.spawn("p", s0, s1);
        p.advance(proc.pid, s1);
        assert_eq!(p.get(proc.pid).unwrap().state, ProcessState::Completed);
    }
}
