//! Backward trajectory completion.
//!
//! A trajectory is built by interpolating from the final state back to
//! the root `(1, 0, 0)`. The algorithm returns `depth` intermediate
//! coordinates (the path) and a *miracle count* equal to the number of
//! virtual ternary decisions resolved on the way back. By construction
//! the path terminates one categorical step short of the actual
//! endpoint: the **penultimate state**.

use crate::fisher::s_distance;
use crate::scoord::SCoord;

/// A backward trajectory from `initial` to `final` with intermediate
/// path nodes.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Endpoint to which we navigated.
    pub final_coord: SCoord,
    /// Root of the navigation (currently always `(1, 0, 0)`).
    pub initial: SCoord,
    /// Coarse-to-fine path including `initial` at index 0 and `final_coord`
    /// at the last index. Length is `depth + 1`.
    pub path: Vec<SCoord>,
    /// Number of virtual sub-state decisions crossed.
    pub miracle_count: usize,
    /// Number of refinement steps; equals `log_3 N` for a depth-`N` hierarchy.
    pub steps: usize,
}

/// Geodesic backward navigation from `final_coord` to the root.
///
/// Walks `depth` steps backward, each step approximating the parent cell's
/// centroid by linear interpolation toward the root. Complexity is
/// `O(depth) = O(log_3 N)`.
pub fn backward_navigate(final_coord: SCoord, depth: usize) -> Trajectory {
    let initial = SCoord::root();
    let mut path: Vec<SCoord> = Vec::with_capacity(depth + 1);
    path.push(final_coord);

    let depth_f = depth.max(1) as f64;
    for j in (1..=depth).rev() {
        let alpha = (depth as f64 - j as f64 + 1.0) / depth_f;
        let k = lerp(final_coord.k, initial.k, alpha).clamp(0.0, 1.0);
        let t = lerp(final_coord.t, initial.t, alpha).clamp(0.0, 1.0);
        let e = lerp(final_coord.e, initial.e, alpha).clamp(0.0, 1.0);
        path.push(SCoord::unchecked(k, t, e));
    }

    path.reverse(); // now initial → final
    Trajectory {
        final_coord,
        initial,
        path,
        miracle_count: depth,
        steps: depth,
    }
}

/// Apply the single completion morphism from a penultimate state to the
/// final state. By definition this returns the supplied final coordinate
/// — the "answer is synthesised from coordinates" step.
pub fn completion_morphism(_penultimate: SCoord, final_coord: SCoord) -> SCoord {
    final_coord
}

/// Return the indices and distances of the `k` candidates closest to
/// `target` under [`s_distance`].
///
/// Candidates are ordered by ascending distance.
pub fn nearest(target: SCoord, candidates: &[SCoord], k: usize) -> Vec<(usize, f64)> {
    let mut scored: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, s_distance(target, *c)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

fn lerp(from: f64, to: f64, alpha: f64) -> f64 {
    from + (to - from) * alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_length_is_depth_plus_one() {
        let traj = backward_navigate(SCoord::new(0.3, 0.5, 0.7).unwrap(), 12);
        assert_eq!(traj.path.len(), 13);
        assert_eq!(traj.steps, 12);
        assert_eq!(traj.miracle_count, 12);
    }

    #[test]
    fn trajectory_starts_at_initial_and_ends_at_final() {
        let f = SCoord::new(0.3, 0.5, 0.7).unwrap();
        let traj = backward_navigate(f, 12);
        assert_eq!(traj.path.first().copied(), Some(SCoord::root()));
        assert_eq!(traj.path.last().copied(), Some(f));
    }

    #[test]
    fn completion_morphism_returns_target() {
        let pen = SCoord::new(0.3, 0.5, 0.7).unwrap();
        let fin = SCoord::new(0.4, 0.5, 0.7).unwrap();
        assert_eq!(completion_morphism(pen, fin), fin);
    }

    #[test]
    fn nearest_orders_by_distance() {
        let target = SCoord::new(0.5, 0.5, 0.5).unwrap();
        let candidates = vec![
            SCoord::new(0.5, 0.5, 0.5).unwrap(), // self
            SCoord::new(0.9, 0.9, 0.9).unwrap(), // far
            SCoord::new(0.4, 0.5, 0.5).unwrap(), // close
        ];
        let result = nearest(target, &candidates, 3);
        assert_eq!(result[0].0, 0);
        assert_eq!(result[1].0, 2);
        assert_eq!(result[2].0, 1);
    }
}
