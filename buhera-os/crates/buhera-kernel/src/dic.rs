//! Demon I/O Controller.
//!
//! Surgical retrieval: given a candidate source and a query coord, return
//! only the top-`k` items closest to the query. Zero-cost categorical
//! sorting: order items by S-distance to the origin.

use buhera_substrate::{s_distance, SCoord};
use serde::{Deserialize, Serialize};

/// One item returned by [`Dic::retrieve`].
#[derive(Debug, Clone)]
pub struct RetrievedItem<T: Clone> {
    /// Coord of the candidate.
    pub coord: SCoord,
    /// Stored value.
    pub value: T,
    /// Distance to the query coord.
    pub distance: f64,
}

/// DIC statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DicStats {
    /// Cumulative items returned across all `retrieve` calls.
    pub bits_retrieved: u64,
    /// Cumulative items available across all `retrieve` calls.
    pub bits_available: u64,
    /// Effective compression `1 - retrieved/available`.
    pub compression: f64,
}

/// Demon I/O Controller.
#[derive(Debug, Default)]
pub struct Dic {
    events: Vec<String>,
    bits_retrieved: u64,
    bits_available: u64,
}

impl Dic {
    /// Construct an empty controller.
    pub fn new() -> Self {
        Self::default()
    }

    /// Surgical retrieval: return the `max_results` items in `source`
    /// closest to `query`.
    pub fn retrieve<T: Clone>(
        &mut self,
        source: &[(SCoord, T)],
        query: SCoord,
        max_results: usize,
    ) -> Vec<RetrievedItem<T>> {
        let mut scored: Vec<RetrievedItem<T>> = source
            .iter()
            .map(|(c, v)| RetrievedItem {
                coord: *c,
                value: v.clone(),
                distance: s_distance(query, *c),
            })
            .collect();
        scored.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(max_results);

        let total = source.len() as u64;
        let returned = scored.len() as u64;
        self.bits_retrieved += returned;
        self.bits_available += total;
        let compression = if total == 0 {
            0.0
        } else {
            1.0 - (returned as f64 / total as f64)
        };
        self.events.push(format!(
            "DIC.retrieve {}/{} ({:.0}% compression)",
            returned,
            total,
            compression * 100.0
        ));
        scored
    }

    /// Zero-cost categorical sort: order items by S-distance to the
    /// origin. By the commutation relation `[O_cat, O_phys] = 0`, this
    /// incurs no thermodynamic work.
    pub fn categorical_sort<T: Clone>(&mut self, items: &[(SCoord, T)]) -> Vec<(SCoord, T)> {
        let origin = SCoord::origin();
        let mut indexed: Vec<(f64, SCoord, T)> = items
            .iter()
            .map(|(c, v)| (s_distance(origin, *c), *c, v.clone()))
            .collect();
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.events
            .push(format!("DIC.categorical_sort {} items W_cat=0", items.len()));
        indexed.into_iter().map(|(_, c, v)| (c, v)).collect()
    }

    /// Cumulative statistics.
    pub fn stats(&self) -> DicStats {
        let compression = if self.bits_available == 0 {
            0.0
        } else {
            1.0 - (self.bits_retrieved as f64 / self.bits_available as f64)
        };
        DicStats {
            bits_retrieved: self.bits_retrieved,
            bits_available: self.bits_available,
            compression,
        }
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
    fn retrieve_top_k() {
        let mut dic = Dic::new();
        let src = vec![
            (SCoord::new(0.1, 0.1, 0.1).unwrap(), "a"),
            (SCoord::new(0.9, 0.9, 0.9).unwrap(), "b"),
            (SCoord::new(0.5, 0.5, 0.5).unwrap(), "c"),
        ];
        let query = SCoord::new(0.55, 0.5, 0.5).unwrap();
        let result = dic.retrieve(&src, query, 1);
        assert_eq!(result[0].value, "c");
    }

    #[test]
    fn categorical_sort_orders_by_origin_distance() {
        let mut dic = Dic::new();
        let items = vec![
            (SCoord::new(0.9, 0.9, 0.9).unwrap(), "far"),
            (SCoord::new(0.1, 0.1, 0.1).unwrap(), "near"),
        ];
        let sorted = dic.categorical_sort(&items);
        assert_eq!(sorted[0].1, "near");
        assert_eq!(sorted[1].1, "far");
    }
}
