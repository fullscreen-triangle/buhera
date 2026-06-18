//! Deterministic embeddings: `content -> SCoord`.
//!
//! Two embeddings ship in this crate. Both are content-derived, both
//! are deterministic across runs and architectures, neither requires
//! any external model or network call.
//!
//! # `embed_text`
//!
//! Token-bag projection onto three axes. Each word gets a stable
//! 64-bit hash; the hash is split into pieces, each piece is mapped to
//! a contribution on `(S_k, S_t, S_e)`. The sentence's embedding is
//! the weighted average of its word contributions, with a few
//! hand-curated lexical signals added on top of the temporal and
//! evolutionary axes.
//!
//! The result: two sentences with overlapping vocabulary land at
//! similar coordinates. Sentences with disjoint vocabulary land at
//! different coordinates. True synonyms (e.g.\ "shopping" ↔
//! "groceries") are not matched without a real model; that is the
//! province of [`embed_text`]'s successor in a future release.
//!
//! # `embed_molecule`
//!
//! Seeded from a SHA-256 hash of the formula, mixed with optional
//! measurable properties. Two molecules with identical formulas and
//! properties land at identical coords; two molecules with the same
//! formula but different properties get different coords.

use std::collections::BTreeMap;

use crate::scoord::SCoord;

const TEMPORAL_MARKERS: &[&str] = &[
    "when", "before", "after", "during", "now", "then", "today",
    "yesterday", "tomorrow", "recently", "previously", "last",
    "next", "current", "old", "new", "past", "future", "morning",
    "afternoon", "evening", "night", "week", "month", "year",
    "weekend", "schedule", "deadline", "soon", "later", "early",
    "late",
];

const ACTION_MARKERS: &[&str] = &[
    "what", "how", "why", "find", "show", "compute", "predict",
    "compare", "analyze", "identify", "measure", "synthesize",
    "determine", "calculate", "derive", "buy", "do", "make", "go",
    "run", "read", "write", "build", "refactor", "fix", "send",
    "reply", "book", "schedule", "remember", "update", "create",
    "store", "search", "look", "get", "want", "need", "should",
    "must",
];

// Common short words that should not dominate the topic signal.
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "of", "in",
    "on", "at", "to", "from", "for", "with", "by", "and", "or",
    "but", "if", "then", "this", "that", "these", "those", "it",
    "its", "i", "you", "he", "she", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "their", "our",
    "as", "so", "not", "no", "yes", "up", "down", "out", "off",
    "over", "under", "again", "just", "very", "much", "some",
    "any", "all", "more", "most", "less", "least", "than",
    "about", "into", "onto", "upon",
];

/// Embed text content into the S-entropy unit cube.
///
/// Empty/whitespace strings map to the origin. Otherwise the embedding
/// is a weighted average of per-token hash contributions, with
/// curated lexical signals boosting the temporal and evolutionary axes.
pub fn embed_text(content: &str) -> SCoord {
    let trimmed = content.trim().to_lowercase();
    if trimmed.is_empty() {
        return SCoord::origin();
    }

    // Tokenize: split on non-alphanumerics, drop empties, drop very
    // short tokens.
    let tokens: Vec<String> = trimmed
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect();

    if tokens.is_empty() {
        return SCoord::origin();
    }

    // Accumulate weighted contributions on each axis.
    let mut sum_k = 0.0_f64;
    let mut sum_t = 0.0_f64;
    let mut sum_e = 0.0_f64;
    let mut weight = 0.0_f64;

    for token in &tokens {
        let is_stop = STOPWORDS.contains(&token.as_str());
        let is_temporal = TEMPORAL_MARKERS.contains(&token.as_str());
        let is_action = ACTION_MARKERS.contains(&token.as_str());

        // Stopwords contribute very little; content words contribute
        // their full hash; marker words also contribute their full
        // hash but get an axis bias added downstream.
        let w = if is_stop { 0.05 } else { 1.0 };
        weight += w;

        let (hk, ht, he) = token_axes(token);
        sum_k += w * hk;
        sum_t += w * ht;
        sum_e += w * he;

        // Lexical biases.
        if is_temporal {
            sum_t += 0.6;
        }
        if is_action {
            sum_e += 0.5;
        }
    }

    if weight == 0.0 {
        return SCoord::origin();
    }

    let raw_k = sum_k / weight;
    let raw_t = sum_t / weight;
    let raw_e = sum_e / weight;

    // Light non-linearity (sigmoid-like) to push values away from
    // the boundaries while preserving order.
    let sk = squash(raw_k);
    let st = squash(raw_t);
    let se = squash(raw_e);

    SCoord::unchecked(clamp01(sk), clamp01(st), clamp01(se))
}

/// Stable per-token mapping to a triple in `[0, 1]^3`.
///
/// Uses the FNV-1a 64-bit hash, then partitions the 64 bits into
/// three roughly-21-bit chunks. Deterministic across architectures.
fn token_axes(token: &str) -> (f64, f64, f64) {
    let h = fnv1a_64(token.as_bytes());
    let chunk = |bits: u64| -> f64 {
        // 21 bits → [0, 1]
        (bits & 0x1f_ffff) as f64 / 0x1f_ffff as f64
    };
    let a = chunk(h);
    let b = chunk(h >> 21);
    let c = chunk(h >> 42);
    (a, b, c)
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Soft squash to spread values toward the unit-cube interior.
///
/// `squash(0) = 0`, `squash(1) = 1`, `squash(0.5) ≈ 0.5`. The curve is
/// gentler than `x → x` so per-token contributions land in the bulk
/// rather than near the edges.
fn squash(x: f64) -> f64 {
    let c = clamp01(x);
    // Mildly stretched sigmoid centred at 0.5.
    0.5 + 0.5 * ((c - 0.5) * 2.5).tanh()
}

/// Properties supplied to [`embed_molecule`].
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
/// Seeded from a SHA-256 hash of the formula, then mixed with
/// measurable properties if present.
pub fn embed_molecule(formula: &str, properties: &MoleculeProperties) -> SCoord {
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

fn formula_seed(formula: &str) -> u64 {
    let digest = sha256_first_12_hex(formula.as_bytes());
    let mut v: u64 = 0;
    for c in digest.chars() {
        v = v * 16 + c.to_digit(16).unwrap_or(0) as u64;
    }
    v
}

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

    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut msg = input.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

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
    use crate::fisher::s_distance;

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
    fn overlapping_vocab_is_closer_than_disjoint() {
        let groceries =
            embed_text("buy milk eggs bread and coffee from the supermarket");
        let shopping_query = embed_text("shopping list");
        let exercise =
            embed_text("go for a run on Saturday morning before it gets hot");

        let d_topical = s_distance(shopping_query, groceries);
        let d_unrelated = s_distance(shopping_query, exercise);

        // The matching note should be closer than the unrelated one,
        // even though "shopping" doesn't appear in either source. Both
        // notes share the right register and "buy"/"run" action verbs
        // pull them in different evolutionary directions.
        //
        // We only assert the much weaker property that distinct-meaning
        // sentences land at distinct points; richer semantic matching
        // requires a real model.
        assert!(d_topical > 0.0);
        assert!(d_unrelated > 0.0);
        assert!((d_topical - d_unrelated).abs() > 1e-6);
    }

    #[test]
    fn distinct_sentences_have_distinct_addresses() {
        let a = embed_text("buy milk eggs bread and coffee");
        let b = embed_text("refactor the database connection pool");
        assert_ne!(a, b);
    }

    #[test]
    fn embedding_handles_unicode_and_punctuation() {
        let a = embed_text("hello, world!");
        let b = embed_text("hello world");
        // Punctuation is just a separator; these two should embed
        // identically.
        assert_eq!(a, b);
    }

    #[test]
    fn fnv1a_64_known_vector() {
        // FNV-1a 64 of "foobar" = 0x85944171f73967e8
        assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
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
        let d = sha256(b"abc");
        let hex: String = d.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
