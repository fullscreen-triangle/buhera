//! Deterministic embeddings: `content -> SCoord`.
//!
//! These mirror `driven/system/substrate.py::embed_text` and
//! `embed_molecule` exactly. Both are content-derived (not learned); the
//! same input yields the same output across runs and architectures.

use std::collections::{BTreeMap, HashMap};

use crate::scoord::SCoord;

const TEMPORAL_MARKERS: &[&str] = &[
    "when", "before", "after", "during", "now", "then",
    "yesterday", "today", "recently", "previously", "last",
    "next", "current", "old", "new", "past", "future",
];

const ACTION_MARKERS: &[&str] = &[
    "what", "how", "why", "find", "show", "compute", "predict",
    "compare", "analyze", "identify", "measure", "synthesize",
    "determine", "calculate", "derive",
];

/// Embed text content into S-entropy space.
///
/// * `S_k`: Shannon entropy over characters, normalised by `log2(26)`.
/// * `S_t`: fraction of words that are temporal markers (scaled).
/// * `S_e`: action-density (question marks + action verbs, scaled).
///
/// The empty string maps to the origin.
pub fn embed_text(content: &str) -> SCoord {
    let trimmed = content.trim().to_lowercase();
    if trimmed.is_empty() {
        return SCoord::origin();
    }

    let chars: Vec<char> = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
    let n = chars.len().max(1) as f64;

    // S_k: Shannon entropy normalised by log2(26)
    let mut freq: HashMap<char, usize> = HashMap::new();
    for c in &chars {
        *freq.entry(*c).or_insert(0) += 1;
    }
    let mut h = 0.0_f64;
    for &count in freq.values() {
        let p = count as f64 / n;
        h -= p * p.log2();
    }
    let sk = (h / 26.0_f64.log2()).min(1.0).max(0.0);

    // Split into words after replacing ? and ! with spaces.
    let space_form: String = trimmed
        .chars()
        .map(|c| if c == '?' || c == '!' { ' ' } else { c })
        .collect();
    let words: Vec<&str> = space_form.split_whitespace().collect();
    let nw = words.len() as f64;

    // S_t: fraction of temporal markers, scaled by 0.3 * |words|.
    let t_hits: f64 = words
        .iter()
        .filter(|w| TEMPORAL_MARKERS.contains(w))
        .count() as f64;
    let st = (t_hits / (nw * 0.3).max(1.0)).min(1.0);

    // S_e: action density (action verbs + '?' count), scaled by 0.4 * |words|.
    let a_hits: f64 = words
        .iter()
        .filter(|w| ACTION_MARKERS.contains(w))
        .count() as f64
        + content.matches('?').count() as f64;
    let se = (a_hits / (nw * 0.4).max(1.0)).min(1.0);

    // Components are mathematically in [0, 1]; clamp to satisfy the
    // SCoord invariant against FP edge cases.
    SCoord::unchecked(clamp01(sk), clamp01(st), clamp01(se))
}

/// Properties supplied to [`embed_molecule`].
///
/// All fields optional. Mirrors the Python implementation, which mixes a
/// formula hash with optional measurable properties.
#[derive(Debug, Clone, Default)]
pub struct MoleculeProperties {
    /// Molecular weight (Da).
    pub molecular_weight: Option<f64>,
    /// Boiling point in degrees Celsius.
    pub boiling_point_c: Option<f64>,
    /// Atom count.
    pub n_atoms: Option<f64>,
}

impl MoleculeProperties {
    /// Build from a JSON-style map (the format used in
    /// `data/nist_compounds.json`).
    pub fn from_map(map: &BTreeMap<String, serde_json::Value>) -> Self {
        Self {
            molecular_weight: map.get("molecular_weight").and_then(|v| v.as_f64()),
            boiling_point_c: map.get("boiling_point_c").and_then(|v| v.as_f64()),
            n_atoms: map.get("n_atoms").and_then(|v| v.as_f64()),
        }
    }
}

/// Embed a molecule into S-entropy space.
///
/// Seeded from a SHA-256 hash of the formula, then mixed with measurable
/// properties if present. Two molecules with identical formulas and
/// properties always map to identical coords; two molecules with the same
/// formula but different properties get different coords.
pub fn embed_molecule(formula: &str, properties: &MoleculeProperties) -> SCoord {
    // The Python uses sha256(formula).hexdigest()[:12] as a 48-bit seed,
    // then three small LCG-style derivations. We mirror that exactly to
    // keep numerical parity with the Python regression oracle.
    let seed = formula_seed(formula);

    let m1: u64 = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
    let m2: u64 = seed.wrapping_mul(1_140_671_485).wrapping_add(12_820_163) & 0x7fff_ffff;
    let m3: u64 = seed.wrapping_mul(214_013).wrapping_add(2_531_011) & 0x7fff_ffff;

    let mut rng_k = m1 as f64 / 0x7fff_ffff_u64 as f64;
    let mut rng_t = m2 as f64 / 0x7fff_ffff_u64 as f64;
    let mut rng_e = m3 as f64 / 0x7fff_ffff_u64 as f64;

    if let Some(mw) = properties.molecular_weight {
        rng_k = (rng_k + (mw + 1.0).ln() / 10.0).rem_euclid(1.0);
    }
    if let Some(bp) = properties.boiling_point_c {
        rng_t = (rng_t + (bp + 273.0) / 1000.0).rem_euclid(1.0);
    }
    if let Some(na) = properties.n_atoms {
        rng_e = (rng_e + (na + 1.0).ln() / 10.0).rem_euclid(1.0);
    }

    SCoord::unchecked(clamp01(rng_k), clamp01(rng_t), clamp01(rng_e))
}

/// Compute a 48-bit seed from the first 12 hex digits of SHA-256(formula),
/// matching the Python `int(hashlib.sha256(formula.encode()).hexdigest()[:12], 16)`.
fn formula_seed(formula: &str) -> u64 {
    let digest = sha256_first_12_hex(formula.as_bytes());
    let mut v: u64 = 0;
    for c in digest.chars() {
        v = v * 16 + c.to_digit(16).unwrap_or(0) as u64;
    }
    v
}

/// Minimal SHA-256 (first 12 hex digits = first 48 bits of the digest).
///
/// We implement it inline rather than pull a crypto dep because:
///   1. The crate has no other dependencies.
///   2. The use is hashing for embedding, not security.
///   3. Parity with Python `hashlib.sha256` is required and well-defined.
fn sha256_first_12_hex(input: &[u8]) -> String {
    let digest = sha256(input);
    let mut s = String::with_capacity(12);
    for b in &digest[..6] {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

// ---------- SHA-256 implementation (FIPS 180-4) ----------

fn sha256(input: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let mut h = [
        0x6a09e667_u32, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Pre-processing: pad input.
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut msg = input.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 64-byte block.
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0]; let mut b = h[1]; let mut c = h[2]; let mut d = h[3];
        let mut e = h[4]; let mut f = h[5]; let mut g = h[6]; let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let mj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(mj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn clamp01(x: f64) -> f64 {
    x.max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_is_origin() {
        assert_eq!(embed_text("").as_tuple(), SCoord::origin().as_tuple());
    }

    #[test]
    fn text_embedding_is_deterministic() {
        let a = embed_text("What is the boiling point of ethanol?");
        let b = embed_text("What is the boiling point of ethanol?");
        assert_eq!(a, b);
    }

    #[test]
    fn text_embedding_is_in_unit_cube() {
        let s = embed_text("The quick brown fox jumps over the lazy dog");
        assert!((0.0..=1.0).contains(&s.k));
        assert!((0.0..=1.0).contains(&s.t));
        assert!((0.0..=1.0).contains(&s.e));
    }

    #[test]
    fn molecule_embedding_is_deterministic() {
        let p = MoleculeProperties {
            molecular_weight: Some(46.07),
            boiling_point_c: Some(78.37),
            n_atoms: Some(9.0),
        };
        let a = embed_molecule("C2H6O", &p);
        let b = embed_molecule("C2H6O", &p);
        assert_eq!(a, b);
    }

    #[test]
    fn molecule_embedding_different_formulas_differ() {
        let p = MoleculeProperties::default();
        let a = embed_molecule("C2H6O", &p);
        let b = embed_molecule("C6H6", &p);
        assert_ne!(a, b);
    }

    #[test]
    fn sha256_known_vector() {
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let d = sha256(b"abc");
        let hex: String = d.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
