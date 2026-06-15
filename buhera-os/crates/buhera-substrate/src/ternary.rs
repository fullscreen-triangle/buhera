//! Ternary categorical addressing.
//!
//! Each address digit refines one of `(k, t, e)` chosen by `position mod 3`.
//! At each step the current interval for the chosen axis is divided into
//! thirds; the digit `0/1/2` records which third contains the coordinate.

use crate::scoord::SCoord;

/// Encode an [`SCoord`] as an interleaved ternary string of length `depth`.
///
/// Digit `3j + 0` refines `k`, `3j + 1` refines `t`, `3j + 2` refines `e`.
/// The result is a string of `0`, `1`, `2` characters.
pub fn ternary_address(s: SCoord, depth: usize) -> String {
    let mut ranges = [(0.0_f64, 1.0_f64); 3];
    let vals = [s.k, s.t, s.e];
    let mut digits = String::with_capacity(depth);

    for d in 0..depth {
        let dim = d % 3;
        let (lo, hi) = ranges[dim];
        let third = (hi - lo) / 3.0;
        let v = vals[dim];
        let trit = if v < lo + third {
            0
        } else if v < lo + 2.0 * third {
            1
        } else {
            2
        };
        digits.push(match trit {
            0 => '0',
            1 => '1',
            _ => '2',
        });
        let f = trit as f64;
        ranges[dim] = (lo + f * third, lo + (f + 1.0) * third);
    }

    digits
}

/// Length of the longest common prefix of two ternary addresses.
pub fn common_prefix_length(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_origin_address() {
        let addr = ternary_address(SCoord::origin(), 6);
        assert_eq!(addr, "000000");
    }

    #[test]
    fn root_address_alternates_appropriately() {
        // root = (1, 0, 0). Digit 0 refines k (=1 -> trit 2);
        // digit 1 refines t (=0 -> 0); digit 2 refines e (=0 -> 0).
        let addr = ternary_address(SCoord::root(), 3);
        assert_eq!(addr, "200");
    }

    #[test]
    fn prefix_of_self_is_full_length() {
        let addr = ternary_address(SCoord::new(0.4, 0.5, 0.6).unwrap(), 9);
        assert_eq!(common_prefix_length(&addr, &addr), addr.len());
    }

    #[test]
    fn near_coords_share_long_prefix() {
        let a = ternary_address(SCoord::new(0.5, 0.5, 0.5).unwrap(), 12);
        let b = ternary_address(SCoord::new(0.5001, 0.5, 0.5).unwrap(), 12);
        // The two coords are extremely close; they should share most
        // initial digits.
        assert!(common_prefix_length(&a, &b) >= 9);
    }
}
