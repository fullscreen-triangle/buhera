//! S-entropy coordinate type.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Construction-time error for [`SCoord`].
#[derive(Debug, Clone, PartialEq)]
pub struct SCoordError {
    /// Coordinate component name (`k`, `t`, or `e`).
    pub component: &'static str,
    /// Out-of-range value supplied.
    pub value: f64,
}

impl fmt::Display for SCoordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SCoord.{}={} outside [0,1]",
            self.component, self.value
        )
    }
}

impl std::error::Error for SCoordError {}

/// A point in S-entropy space `[0,1]^3`.
///
/// * `k` — knowledge (information deficit).
/// * `t` — temporal (position in the completion sequence).
/// * `e` — entropy (constraint density).
///
/// Coordinates are validated at construction to lie in `[0, 1]` modulo a
/// `1e-9` tolerance for floating-point round-off (matching the Python
/// reference).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SCoord {
    /// Knowledge axis.
    pub k: f64,
    /// Temporal axis.
    pub t: f64,
    /// Entropy axis.
    pub e: f64,
}

const TOL: f64 = 1e-9;

impl SCoord {
    /// Construct an `SCoord`, returning an error if any component lies
    /// outside `[0, 1]`.
    pub fn new(k: f64, t: f64, e: f64) -> Result<Self, SCoordError> {
        for (name, v) in [("k", k), ("t", t), ("e", e)] {
            if !(-TOL..=1.0 + TOL).contains(&v) {
                return Err(SCoordError { component: name, value: v });
            }
        }
        Ok(Self { k, t, e })
    }

    /// Construct an `SCoord` without bounds checking. Used in internal hot
    /// paths after a clamp; never accept user input through this.
    pub(crate) fn unchecked(k: f64, t: f64, e: f64) -> Self {
        Self { k, t, e }
    }

    /// Tuple view.
    pub fn as_tuple(&self) -> (f64, f64, f64) {
        (self.k, self.t, self.e)
    }

    /// Origin `(0, 0, 0)`.
    pub fn origin() -> Self {
        Self { k: 0.0, t: 0.0, e: 0.0 }
    }

    /// Root `(1, 0, 0)` — the "maximum uncertainty" point from which
    /// processes spawn.
    pub fn root() -> Self {
        Self { k: 1.0, t: 0.0, e: 0.0 }
    }
}

impl fmt::Display for SCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S({:.3},{:.3},{:.3})", self.k, self.t, self.e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_coord_succeeds() {
        assert!(SCoord::new(0.0, 0.5, 1.0).is_ok());
    }

    #[test]
    fn out_of_range_fails() {
        let err = SCoord::new(1.5, 0.0, 0.0).unwrap_err();
        assert_eq!(err.component, "k");
    }

    #[test]
    fn tolerance_accepts_tiny_overshoot() {
        assert!(SCoord::new(-1e-10, 1.0 + 1e-10, 0.5).is_ok());
    }

    #[test]
    fn root_and_origin() {
        assert_eq!(SCoord::root().as_tuple(), (1.0, 0.0, 0.0));
        assert_eq!(SCoord::origin().as_tuple(), (0.0, 0.0, 0.0));
    }
}
