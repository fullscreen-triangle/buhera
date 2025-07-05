//! Mathematical foundations and computational primitives
//!
//! This module provides mathematical foundations for the Buhera framework.

use crate::error::BuheraResult;

/// Mathematical constants and functions for the Buhera framework
pub mod constants {
    /// Planck's constant
    pub const PLANCK_CONSTANT: f64 = 6.62607015e-34;
    
    /// Boltzmann constant
    pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;
    
    /// Default fuzzy precision
    pub const DEFAULT_FUZZY_PRECISION: f64 = 0.001;
}

/// Calculate quantum coherence time
pub fn calculate_coherence_time(temperature: f64) -> f64 {
    constants::PLANCK_CONSTANT / (constants::BOLTZMANN_CONSTANT * temperature)
}

/// Calculate fuzzy membership function
pub fn triangular_membership(x: f64, a: f64, b: f64, c: f64) -> f64 {
    if x <= a || x >= c {
        0.0
    } else if x <= b {
        (x - a) / (b - a)
    } else {
        (c - x) / (c - b)
    }
} 