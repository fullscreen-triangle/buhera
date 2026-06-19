//! Token-overlap scoring (the option-C safety net).
//!
//! After the kernel returns its `k` nearest matches by S-distance, the
//! REPL/demo re-rank them slightly by counting literal token overlap
//! with the query. This rescues short queries whose semantic
//! embedding lacks enough signal to disambiguate, and queries whose
//! target literally contains the query words.

use std::collections::HashSet;

/// Compute a token-overlap score in `[0, 1]`.
///
/// * `0.0` — no shared tokens.
/// * `1.0` — every query token appears in the target.
///
/// Tokens are split on non-alphanumerics, lowercased, and filtered to
/// length ≥ 3 to suppress common stopwords.
pub fn token_overlap_score(query: &str, target: &str) -> f64 {
    let q_tokens = tokenize(query);
    if q_tokens.is_empty() {
        return 0.0;
    }
    let t_tokens = tokenize(target);
    if t_tokens.is_empty() {
        return 0.0;
    }
    let hit = q_tokens
        .iter()
        .filter(|tok| t_tokens.contains(*tok))
        .count();
    hit as f64 / q_tokens.len() as f64
}

fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= 3)
        .map(|s| s.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_overlap_is_zero() {
        assert!(token_overlap_score("apple banana", "carrot dragonfruit") < 1e-9);
    }

    #[test]
    fn full_overlap_is_one() {
        let s = token_overlap_score("apple banana", "apple banana cherry");
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn partial_overlap_is_fractional() {
        let s = token_overlap_score("morning workout", "Saturday morning run");
        assert!((s - 0.5).abs() < 1e-9);
    }

    #[test]
    fn case_insensitive() {
        assert!(token_overlap_score("APPLE", "apple sauce") > 0.0);
    }

    #[test]
    fn short_tokens_are_ignored() {
        // "the" is too short (len < 3? actually len = 3 = threshold).
        // Tokens shorter than 3 are filtered.
        let s = token_overlap_score("a b c", "x y z");
        assert!(s < 1e-9);
    }
}
