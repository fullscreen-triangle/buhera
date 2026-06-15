//! Fisher information metric on `[0,1]^3`.
//!
//! The 1-D Fisher distance on `(0, 1)` with metric `ds² = dx² / (x(1-x))`
//! has the closed form
//!
//! ```text
//! d(a, b) = | asin(2a - 1) - asin(2b - 1) |
//! ```
//!
//! up to clamping at the endpoints. The 3-D distance is the Euclidean
//! sum of the three per-axis distances (product Fisher metric).

use crate::scoord::SCoord;

const EPS: f64 = 1e-6;

/// Geodesic distance on `(0, 1)` under the Fisher metric.
///
/// Endpoints are clamped to `[EPS, 1 - EPS]` to avoid the singularity at
/// `x = 0` and `x = 1`.
pub fn fisher_distance_1d(a: f64, b: f64) -> f64 {
    let a = a.max(EPS).min(1.0 - EPS);
    let b = b.max(EPS).min(1.0 - EPS);
    ((2.0 * a - 1.0).asin() - (2.0 * b - 1.0).asin()).abs()
}

/// Product-Fisher distance on `[0,1]^3`.
pub fn s_distance(s1: SCoord, s2: SCoord) -> f64 {
    let dk = fisher_distance_1d(s1.k, s2.k);
    let dt = fisher_distance_1d(s1.t, s2.t);
    let de = fisher_distance_1d(s1.e, s2.e);
    (dk * dk + dt * dt + de * de).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_distance_is_zero() {
        let s = SCoord::new(0.3, 0.5, 0.7).unwrap();
        assert!(s_distance(s, s) < 1e-12);
    }

    #[test]
    fn symmetric() {
        let a = SCoord::new(0.2, 0.4, 0.6).unwrap();
        let b = SCoord::new(0.7, 0.1, 0.9).unwrap();
        assert!((s_distance(a, b) - s_distance(b, a)).abs() < 1e-12);
    }

    #[test]
    fn triangle_inequality() {
        let a = SCoord::new(0.1, 0.2, 0.3).unwrap();
        let b = SCoord::new(0.4, 0.5, 0.6).unwrap();
        let c = SCoord::new(0.7, 0.8, 0.9).unwrap();
        assert!(s_distance(a, c) <= s_distance(a, b) + s_distance(b, c) + 1e-12);
    }

    #[test]
    fn endpoints_do_not_panic() {
        // Either argument at 0 or 1 hits the EPS clamp.
        let zero = fisher_distance_1d(0.0, 0.5);
        let one = fisher_distance_1d(1.0, 0.5);
        assert!(zero.is_finite());
        assert!(one.is_finite());
    }
}
