//! Deterministic 384 → 3 projection.
//!
//! Three fixed orthogonal directions seed three projections; the dot
//! product of the input vector with each direction is squashed through
//! a tanh-based sigmoid to land in `[0, 1]`.
//!
//! The projection vectors are derived from a stable PRNG seeded with
//! axis-specific bytes, so they are identical across runs and
//! architectures.

use buhera_substrate::SCoord;

/// Hidden dimension of MiniLM and BGE-small models.
pub(crate) const HIDDEN_DIM: usize = 384;

/// Three deterministic unit vectors, one per S-entropy axis.
pub(crate) struct Projection {
    pub(crate) axis_k: Vec<f32>,
    pub(crate) axis_t: Vec<f32>,
    pub(crate) axis_e: Vec<f32>,
}

impl Projection {
    /// Construct the default projection.
    pub(crate) fn default() -> Self {
        Self {
            axis_k: stable_unit_vector(b"buhera-os::axis::S_k", HIDDEN_DIM),
            axis_t: stable_unit_vector(b"buhera-os::axis::S_t", HIDDEN_DIM),
            axis_e: stable_unit_vector(b"buhera-os::axis::S_e", HIDDEN_DIM),
        }
    }

    /// Project a 384-D vector into `[0, 1]^3`.
    pub(crate) fn project(&self, v: &[f32]) -> SCoord {
        assert_eq!(v.len(), HIDDEN_DIM, "embedding has wrong dimension");
        let k_raw = dot(v, &self.axis_k);
        let t_raw = dot(v, &self.axis_t);
        let e_raw = dot(v, &self.axis_e);
        SCoord {
            k: squash(k_raw) as f64,
            t: squash(t_raw) as f64,
            e: squash(e_raw) as f64,
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Squash an arbitrary real to `[0, 1]` via a tanh-based sigmoid.
///
/// Centred at 0, slope tuned so that typical dot products of an
/// L2-normalised 384-D embedding (which fall in roughly `[-1, 1]`) span
/// the unit interval without saturating at the edges.
fn squash(x: f32) -> f32 {
    0.5 + 0.5 * (1.5 * x).tanh()
}

/// Generate a deterministic unit vector of length `dim`.
fn stable_unit_vector(seed: &[u8], dim: usize) -> Vec<f32> {
    // SplitMix-style PRNG seeded from a FNV-1a hash of `seed`.
    let mut state = fnv1a_64(seed).wrapping_add(0xdeadbeef_cafe_babe);
    let mut v: Vec<f32> = Vec::with_capacity(dim);

    // Box-Muller pairs from uniform draws → standard normal samples.
    while v.len() < dim {
        let u1 = (next_u64(&mut state) as f64 / u64::MAX as f64).max(1e-10);
        let u2 = next_u64(&mut state) as f64 / u64::MAX as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        v.push((r * theta.cos()) as f32);
        if v.len() < dim {
            v.push((r * theta.sin()) as f32);
        }
    }
    v.truncate(dim);

    // Normalise to unit length.
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn next_u64(state: &mut u64) -> u64 {
    // SplitMix64.
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_is_deterministic() {
        let p1 = Projection::default();
        let p2 = Projection::default();
        assert_eq!(p1.axis_k, p2.axis_k);
        assert_eq!(p1.axis_t, p2.axis_t);
        assert_eq!(p1.axis_e, p2.axis_e);
    }

    #[test]
    fn projection_axes_are_unit() {
        let p = Projection::default();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm(&p.axis_k) - 1.0).abs() < 1e-4);
        assert!((norm(&p.axis_t) - 1.0).abs() < 1e-4);
        assert!((norm(&p.axis_e) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn projection_axes_are_distinct() {
        let p = Projection::default();
        assert_ne!(p.axis_k, p.axis_t);
        assert_ne!(p.axis_t, p.axis_e);
        assert_ne!(p.axis_k, p.axis_e);
    }

    #[test]
    fn squash_bounds() {
        assert!(squash(-100.0) >= 0.0);
        assert!(squash(100.0) <= 1.0);
        assert!((squash(0.0) - 0.5).abs() < 1e-6);
    }
}
