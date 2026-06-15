//! Categorical Memory Manager.
//!
//! Maps S-entropy coordinates to ternary addresses and tier labels
//! (`L1/L2/L3/RAM`), then stores opaque payloads under those addresses.
//! Proximity queries return the `k` nearest objects under
//! [`buhera_substrate::s_distance`].

use std::collections::BTreeMap;

use buhera_substrate::{s_distance, ternary_address, SCoord};
use serde::{Deserialize, Serialize};

const TIER_THRESHOLDS: [f64; 3] = [0.5, 1.0, 1.5];

/// Tier label for a stored object, derived from `||S||_2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tier {
    /// Hottest tier (largest norm).
    L1,
    /// Next tier.
    L2,
    /// Cooler tier.
    L3,
    /// Coldest tier (smallest norm).
    Ram,
}

impl Tier {
    fn for_coord(s: SCoord) -> Self {
        let n = (s.k * s.k + s.t * s.t + s.e * s.e).sqrt();
        if n > TIER_THRESHOLDS[0] {
            Tier::L1
        } else if n > TIER_THRESHOLDS[1] {
            Tier::L2
        } else if n > TIER_THRESHOLDS[2] {
            Tier::L3
        } else {
            Tier::Ram
        }
    }

    /// String tag matching the Python convention.
    pub fn as_str(&self) -> &'static str {
        match self {
            Tier::L1 => "L1",
            Tier::L2 => "L2",
            Tier::L3 => "L3",
            Tier::Ram => "RAM",
        }
    }
}

/// One memory object: a payload + categorical coordinate + ternary address.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryObject {
    /// S-entropy coordinate.
    pub coord: SCoord,
    /// Ternary address (depth-`Cmm::depth` string of `0/1/2`).
    pub address: String,
    /// Tier classification.
    pub tier: Tier,
    /// Opaque payload (e.g. a NIST property dictionary).
    pub payload: serde_json::Value,
    /// Arbitrary metadata.
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Categorical Memory Manager.
#[derive(Debug, Clone)]
pub struct Cmm {
    /// Depth used to build ternary addresses.
    pub depth: usize,
    store: BTreeMap<String, MemoryObject>,
    events: Vec<String>,
}

impl Cmm {
    /// Construct a CMM with the given ternary-address depth.
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            store: BTreeMap::new(),
            events: Vec::new(),
        }
    }

    /// Allocate an object at `coord`.
    pub fn allocate(
        &mut self,
        coord: SCoord,
        payload: serde_json::Value,
        metadata: BTreeMap<String, serde_json::Value>,
    ) -> MemoryObject {
        let address = ternary_address(coord, self.depth);
        let obj = MemoryObject {
            coord,
            address: address.clone(),
            tier: Tier::for_coord(coord),
            payload,
            metadata,
        };
        self.events
            .push(format!("CMM.allocate addr={} tier={}", address, obj.tier.as_str()));
        self.store.insert(address, obj.clone());
        obj
    }

    /// Look up an object by its address.
    pub fn lookup(&self, address: &str) -> Option<&MemoryObject> {
        self.store.get(address)
    }

    /// Return the `k` objects closest to `target` in S-distance.
    pub fn proximity_query(&self, target: SCoord, k: usize) -> Vec<(MemoryObject, f64)> {
        let mut scored: Vec<(MemoryObject, f64)> = self
            .store
            .values()
            .map(|obj| (obj.clone(), s_distance(target, obj.coord)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// All stored objects, in address order.
    pub fn all_objects(&self) -> Vec<MemoryObject> {
        self.store.values().cloned().collect()
    }

    /// Number of stored objects.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Activity events emitted by this subsystem.
    pub fn events(&self) -> Vec<String> {
        self.events.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn empty_metadata() -> BTreeMap<String, serde_json::Value> {
        BTreeMap::new()
    }

    #[test]
    fn allocate_and_lookup() {
        let mut cmm = Cmm::new(6);
        let s = SCoord::new(0.5, 0.5, 0.5).unwrap();
        let obj = cmm.allocate(s, json!({"v": 1}), empty_metadata());
        let found = cmm.lookup(&obj.address).expect("missing");
        assert_eq!(found.coord, s);
    }

    #[test]
    fn proximity_query_orders_by_distance() {
        let mut cmm = Cmm::new(6);
        let a = cmm.allocate(SCoord::new(0.1, 0.1, 0.1).unwrap(), json!({}), empty_metadata());
        let _b = cmm.allocate(SCoord::new(0.9, 0.9, 0.9).unwrap(), json!({}), empty_metadata());
        let target = SCoord::new(0.1, 0.1, 0.15).unwrap();
        let result = cmm.proximity_query(target, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0.address, a.address);
    }
}
