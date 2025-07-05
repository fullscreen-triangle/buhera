//! Molecular substrate interface and management
//!
//! This module handles molecular-scale computational substrates for the Buhera framework.

use crate::error::BuheraResult;

/// Molecular substrate manager
#[derive(Debug)]
pub struct MolecularSubstrate {
    /// Substrate configuration
    pub config: SubstrateConfig,
}

/// Substrate configuration
#[derive(Debug, Clone)]
pub struct SubstrateConfig {
    /// Substrate type
    pub substrate_type: String,
    /// Temperature in Celsius
    pub temperature: f64,
    /// pH level
    pub ph: f64,
    /// Ionic strength
    pub ionic_strength: f64,
}

impl MolecularSubstrate {
    /// Create a new molecular substrate
    pub fn new(config: SubstrateConfig) -> BuheraResult<Self> {
        Ok(Self { config })
    }

    /// Create synthetic biology configuration
    pub fn synthetic_biology_config() -> BuheraResult<Self> {
        let config = SubstrateConfig {
            substrate_type: "synthetic_biology".to_string(),
            temperature: 37.0,
            ph: 7.4,
            ionic_strength: 150.0,
        };
        Self::new(config)
    }
}

impl Default for MolecularSubstrate {
    fn default() -> Self {
        Self::synthetic_biology_config().expect("Failed to create default molecular substrate")
    }
} 