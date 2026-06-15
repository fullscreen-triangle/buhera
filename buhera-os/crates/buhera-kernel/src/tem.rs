//! Triple Equivalence Monitor.
//!
//! At each kernel tick, the TEM samples the current categorical
//! coordinate and computes three "equivalent entropies" — oscillator,
//! categorical, partition — that must agree within a relative threshold.
//! Divergences are surfaced as alerts.

use buhera_substrate::SCoord;
use serde::{Deserialize, Serialize};

/// One TEM sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemSample {
    /// Coordinate at the sample.
    pub coord: SCoord,
    /// Oscillator-form entropy.
    pub s_osc: f64,
    /// Categorical-form entropy.
    pub s_cat: f64,
    /// Partition-form entropy.
    pub s_par: f64,
    /// Maximum normalised pairwise gap.
    pub delta: f64,
    /// Free-text description.
    pub description: String,
}

/// TEM statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemStats {
    /// Total samples observed.
    pub samples: u64,
    /// Total alerts raised.
    pub alerts: u64,
    /// Largest `delta` seen.
    pub max_delta: f64,
}

/// Triple Equivalence Monitor.
#[derive(Debug)]
pub struct Tem {
    threshold: f64,
    samples: Vec<TemSample>,
    alerts: Vec<String>,
}

impl Tem {
    /// Construct a TEM with the given relative-divergence threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            samples: Vec::new(),
            alerts: Vec::new(),
        }
    }

    /// Construct a TEM with the default threshold (`1e-3`).
    pub fn with_default_threshold() -> Self {
        Self::new(1e-3)
    }

    /// Record a sample at `coord`; raise an alert if the three entropy
    /// forms diverge by more than `threshold` (relative to their max).
    pub fn sample(&mut self, coord: SCoord, description: &str) {
        let (k, t, e) = (coord.k, coord.t, coord.e);

        // Oscillator-form: -Σ v ln v (with a small epsilon to avoid
        // log(0)).
        let eps = 1e-9_f64;
        let s_osc = -((k + eps) * (k + eps).ln()
            + (t + eps) * (t + eps).ln()
            + (e + eps) * (e + eps).ln());

        // Categorical-form: mean * ln 3.
        let s_cat = (k + t + e) / 3.0 * 3.0_f64.ln();

        // Partition-form: ||S||_2 * ln 3.
        let s_par = (k * k + t * t + e * e).sqrt() * 3.0_f64.ln();

        let max_s = s_osc.max(s_cat).max(s_par).max(1e-9);
        let d_osc_cat = (s_osc - s_cat).abs();
        let d_cat_par = (s_cat - s_par).abs();
        let d_osc_par = (s_osc - s_par).abs();
        let delta = d_osc_cat.max(d_cat_par).max(d_osc_par) / max_s;

        self.samples.push(TemSample {
            coord,
            s_osc,
            s_cat,
            s_par,
            delta,
            description: description.to_string(),
        });

        if delta > self.threshold {
            self.alerts
                .push(format!("TEM alert: delta={:.4} at {}", delta, description));
        }
    }

    /// Cumulative statistics.
    pub fn stats(&self) -> TemStats {
        let max_delta = self
            .samples
            .iter()
            .map(|s| s.delta)
            .fold(0.0_f64, f64::max);
        TemStats {
            samples: self.samples.len() as u64,
            alerts: self.alerts.len() as u64,
            max_delta,
        }
    }

    /// Alert events.
    pub fn events(&self) -> Vec<String> {
        self.alerts.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_records_and_counts() {
        let mut tem = Tem::with_default_threshold();
        tem.sample(SCoord::new(0.4, 0.5, 0.6).unwrap(), "first");
        assert_eq!(tem.stats().samples, 1);
    }
}
