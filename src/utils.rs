//! Utilities and helper functions
//!
//! This module provides utility functions for the Buhera framework.

use crate::error::BuheraResult;
use std::time::{SystemTime, UNIX_EPOCH};

/// Generate a unique identifier
pub fn generate_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("buhera_{}", timestamp)
}

/// Validate fuzzy value (must be between 0 and 1)
pub fn validate_fuzzy_value(value: f64) -> BuheraResult<()> {
    if value < 0.0 || value > 1.0 {
        Err(crate::error::FuzzyError::InvalidFuzzyValue { value }.into())
    } else {
        Ok(())
    }
}

/// Calculate entropy
pub fn calculate_entropy(probabilities: &[f64]) -> f64 {
    -probabilities.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.log2())
        .sum()
} 