//! # Kinetic Theory Calculator
//!
//! This module implements the fundamental temperature-oscillation relationship
//! based on kinetic theory of gases. The key insight is that molecular
//! oscillation frequency is proportional to the square root of temperature.
//!
//! ## Key Relationship
//!
//! ```
//! Average Kinetic Energy = (3/2)kT
//! Oscillation Frequency ∝ √T
//! Higher Temperature → Faster Oscillations → Higher Precision
//! ```

use super::{MoleculeType, OscillationFrequency, ThermodynamicConfig, ThermodynamicError, ThermodynamicResult};
use std::collections::HashMap;
use std::f64::consts::PI;
use serde::{Deserialize, Serialize};

/// Kinetic theory calculator for temperature-oscillation relationships
pub struct KineticTheoryCalculator {
    /// Boltzmann constant (J/K)
    k_boltzmann: f64,
    /// Molecular mass database (kg)
    molecular_masses: HashMap<MoleculeType, f64>,
    /// Temperature-frequency conversion factors
    conversion_factors: HashMap<MoleculeType, f64>,
    /// Configuration reference
    config: ThermodynamicConfig,
}

/// Velocity distribution data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VelocityDistribution {
    /// Most probable velocity (m/s)
    pub most_probable_velocity: f64,
    /// Average velocity (m/s)
    pub average_velocity: f64,
    /// RMS velocity (m/s)
    pub rms_velocity: f64,
    /// Temperature (K)
    pub temperature: f64,
    /// Molecular mass (kg)
    pub molecular_mass: f64,
}

/// Molecular vibration modes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VibrationModes {
    /// Symmetric stretching frequency (Hz)
    pub symmetric_stretching: f64,
    /// Asymmetric stretching frequency (Hz)
    pub asymmetric_stretching: f64,
    /// Bending frequency (Hz)
    pub bending: f64,
    /// Rotational frequency (Hz)
    pub rotational: f64,
    /// Translational frequency (Hz)
    pub translational: f64,
}

impl KineticTheoryCalculator {
    /// Create a new kinetic theory calculator
    pub fn new(config: &ThermodynamicConfig) -> ThermodynamicResult<Self> {
        let mut molecular_masses = HashMap::new();
        let mut conversion_factors = HashMap::new();
        
        // Initialize molecular masses (kg)
        molecular_masses.insert(MoleculeType::N2, 4.65e-26);   // Nitrogen
        molecular_masses.insert(MoleculeType::O2, 5.31e-26);   // Oxygen
        molecular_masses.insert(MoleculeType::H2O, 2.99e-26);  // Water
        molecular_masses.insert(MoleculeType::He, 6.64e-27);   // Helium
        molecular_masses.insert(MoleculeType::Ne, 3.35e-26);   // Neon
        molecular_masses.insert(MoleculeType::Ar, 6.63e-26);   // Argon
        
        // Initialize conversion factors for oscillation frequency
        conversion_factors.insert(MoleculeType::N2, 1.2e11);   // Nitrogen
        conversion_factors.insert(MoleculeType::O2, 1.1e11);   // Oxygen
        conversion_factors.insert(MoleculeType::H2O, 1.5e11);  // Water
        conversion_factors.insert(MoleculeType::He, 2.0e11);   // Helium
        conversion_factors.insert(MoleculeType::Ne, 1.3e11);   // Neon
        conversion_factors.insert(MoleculeType::Ar, 1.0e11);   // Argon
        
        Ok(Self {
            k_boltzmann: 1.380649e-23, // J/K
            molecular_masses,
            conversion_factors,
            config: config.clone(),
        })
    }
    
    /// Calculate oscillation frequency from temperature
    pub async fn calculate_oscillation_frequency(
        &self,
        temperature: f64,
        molecule_type: &MoleculeType,
    ) -> ThermodynamicResult<OscillationFrequency> {
        // Get molecular mass
        let molecular_mass = self.molecular_masses.get(molecule_type)
            .ok_or_else(|| ThermodynamicError::Calculation {
                message: format!("Unknown molecule type: {:?}", molecule_type),
            })?;
        
        // Calculate average kinetic energy
        let avg_kinetic_energy = 1.5 * self.k_boltzmann * temperature;
        
        // Calculate RMS velocity
        let rms_velocity = (3.0 * self.k_boltzmann * temperature / molecular_mass).sqrt();
        
        // Calculate oscillation frequency using temperature-frequency relationship
        let frequency = self.calculate_frequency_from_temperature(temperature, molecule_type)?;
        
        Ok(OscillationFrequency {
            frequency,
            temperature,
            molecular_mass: *molecular_mass,
            rms_velocity,
            kinetic_energy: avg_kinetic_energy,
        })
    }
    
    /// Calculate velocity distribution
    pub async fn calculate_velocity_distribution(
        &self,
        temperature: f64,
        molecule_type: &MoleculeType,
    ) -> ThermodynamicResult<VelocityDistribution> {
        let molecular_mass = self.molecular_masses.get(molecule_type)
            .ok_or_else(|| ThermodynamicError::Calculation {
                message: format!("Unknown molecule type: {:?}", molecule_type),
            })?;
        
        // Most probable velocity
        let most_probable_velocity = (2.0 * self.k_boltzmann * temperature / molecular_mass).sqrt();
        
        // Average velocity
        let average_velocity = (8.0 * self.k_boltzmann * temperature / (PI * molecular_mass)).sqrt();
        
        // RMS velocity
        let rms_velocity = (3.0 * self.k_boltzmann * temperature / molecular_mass).sqrt();
        
        Ok(VelocityDistribution {
            most_probable_velocity,
            average_velocity,
            rms_velocity,
            temperature,
            molecular_mass: *molecular_mass,
        })
    }
    
    /// Calculate molecular vibration modes
    pub async fn calculate_vibration_modes(
        &self,
        temperature: f64,
        pressure: f64,
        molecule_type: &MoleculeType,
    ) -> ThermodynamicResult<VibrationModes> {
        let base_frequency = self.calculate_frequency_from_temperature(temperature, molecule_type)?;
        
        // Calculate different vibration modes based on molecular structure
        let modes = match molecule_type {
            MoleculeType::N2 => VibrationModes {
                symmetric_stretching: base_frequency * 2.0,
                asymmetric_stretching: base_frequency * 1.8,
                bending: base_frequency * 0.5,
                rotational: base_frequency * 0.1,
                translational: base_frequency * 0.01,
            },
            MoleculeType::O2 => VibrationModes {
                symmetric_stretching: base_frequency * 1.9,
                asymmetric_stretching: base_frequency * 1.7,
                bending: base_frequency * 0.4,
                rotational: base_frequency * 0.12,
                translational: base_frequency * 0.012,
            },
            MoleculeType::H2O => VibrationModes {
                symmetric_stretching: base_frequency * 3.0,
                asymmetric_stretching: base_frequency * 2.8,
                bending: base_frequency * 1.2,
                rotational: base_frequency * 0.3,
                translational: base_frequency * 0.03,
            },
            MoleculeType::He => VibrationModes {
                symmetric_stretching: base_frequency * 1.0,
                asymmetric_stretching: base_frequency * 1.0,
                bending: 0.0, // Monatomic
                rotational: 0.0, // Monatomic
                translational: base_frequency * 0.1,
            },
            MoleculeType::Ne => VibrationModes {
                symmetric_stretching: base_frequency * 1.0,
                asymmetric_stretching: base_frequency * 1.0,
                bending: 0.0, // Monatomic
                rotational: 0.0, // Monatomic
                translational: base_frequency * 0.08,
            },
            MoleculeType::Ar => VibrationModes {
                symmetric_stretching: base_frequency * 1.0,
                asymmetric_stretching: base_frequency * 1.0,
                bending: 0.0, // Monatomic
                rotational: 0.0, // Monatomic
                translational: base_frequency * 0.05,
            },
            MoleculeType::Custom(_) => VibrationModes {
                symmetric_stretching: base_frequency * 1.5,
                asymmetric_stretching: base_frequency * 1.3,
                bending: base_frequency * 0.6,
                rotational: base_frequency * 0.15,
                translational: base_frequency * 0.015,
            },
        };
        
        // Apply pressure correction
        let pressure_factor = 1.0 + (pressure - 1.0) * 0.1; // Small pressure dependence
        
        Ok(VibrationModes {
            symmetric_stretching: modes.symmetric_stretching * pressure_factor,
            asymmetric_stretching: modes.asymmetric_stretching * pressure_factor,
            bending: modes.bending * pressure_factor,
            rotational: modes.rotational * pressure_factor,
            translational: modes.translational * pressure_factor,
        })
    }
    
    /// Calculate temperature from oscillation frequency (inverse calculation)
    pub async fn calculate_temperature_from_frequency(
        &self,
        frequency: f64,
        molecule_type: &MoleculeType,
    ) -> ThermodynamicResult<f64> {
        let conversion_factor = self.conversion_factors.get(molecule_type)
            .ok_or_else(|| ThermodynamicError::Calculation {
                message: format!("Unknown molecule type: {:?}", molecule_type),
            })?;
        
        // Inverse of frequency-temperature relationship
        let temperature = (frequency / conversion_factor).powi(2);
        
        Ok(temperature)
    }
    
    /// Calculate precision improvement from temperature increase
    pub async fn calculate_precision_improvement(
        &self,
        temperature_increase: f64,
        base_temperature: f64,
        molecule_type: &MoleculeType,
    ) -> ThermodynamicResult<f64> {
        let base_frequency = self.calculate_frequency_from_temperature(base_temperature, molecule_type)?;
        let new_frequency = self.calculate_frequency_from_temperature(
            base_temperature + temperature_increase,
            molecule_type,
        )?;
        
        // Precision improvement is proportional to frequency increase
        let precision_improvement = new_frequency / base_frequency;
        
        Ok(precision_improvement)
    }
    
    /// Calculate frequency from temperature using empirical relationship
    fn calculate_frequency_from_temperature(
        &self,
        temperature: f64,
        molecule_type: &MoleculeType,
    ) -> ThermodynamicResult<f64> {
        let conversion_factor = self.conversion_factors.get(molecule_type)
            .ok_or_else(|| ThermodynamicError::Calculation {
                message: format!("Unknown molecule type: {:?}", molecule_type),
            })?;
        
        // Fundamental relationship: frequency ∝ √T
        let frequency = conversion_factor * temperature.sqrt();
        
        Ok(frequency)
    }
    
    /// Calculate effective temperature from molecular mixture
    pub async fn calculate_effective_temperature(
        &self,
        molecular_mixture: &[(MoleculeType, f64)],
        individual_temperatures: &HashMap<MoleculeType, f64>,
    ) -> ThermodynamicResult<f64> {
        let mut weighted_temperature = 0.0;
        let mut total_weight = 0.0;
        
        for (molecule_type, fraction) in molecular_mixture {
            let temperature = individual_temperatures.get(molecule_type)
                .ok_or_else(|| ThermodynamicError::Calculation {
                    message: format!("Temperature not provided for molecule: {:?}", molecule_type),
                })?;
            
            let molecular_mass = self.molecular_masses.get(molecule_type)
                .ok_or_else(|| ThermodynamicError::Calculation {
                    message: format!("Unknown molecule type: {:?}", molecule_type),
                })?;
            
            // Weight by molecular mass and fraction
            let weight = fraction * molecular_mass;
            weighted_temperature += temperature * weight;
            total_weight += weight;
        }
        
        if total_weight == 0.0 {
            return Err(ThermodynamicError::Calculation {
                message: "Total molecular weight is zero".to_string(),
            });
        }
        
        Ok(weighted_temperature / total_weight)
    }
    
    /// Calculate thermal velocity spread
    pub async fn calculate_thermal_velocity_spread(
        &self,
        temperature: f64,
        molecule_type: &MoleculeType,
    ) -> ThermodynamicResult<f64> {
        let molecular_mass = self.molecular_masses.get(molecule_type)
            .ok_or_else(|| ThermodynamicError::Calculation {
                message: format!("Unknown molecule type: {:?}", molecule_type),
            })?;
        
        // Calculate standard deviation of velocity distribution
        let velocity_spread = (self.k_boltzmann * temperature / molecular_mass).sqrt();
        
        Ok(velocity_spread)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_oscillation_frequency_calculation() {
        let config = ThermodynamicConfig::default();
        let calculator = KineticTheoryCalculator::new(&config).unwrap();
        
        let frequency = calculator.calculate_oscillation_frequency(300.0, &MoleculeType::N2).await.unwrap();
        assert!(frequency.frequency > 0.0);
        assert_eq!(frequency.temperature, 300.0);
        assert!(frequency.rms_velocity > 0.0);
        assert!(frequency.kinetic_energy > 0.0);
    }

    #[test]
    async fn test_velocity_distribution() {
        let config = ThermodynamicConfig::default();
        let calculator = KineticTheoryCalculator::new(&config).unwrap();
        
        let distribution = calculator.calculate_velocity_distribution(300.0, &MoleculeType::N2).await.unwrap();
        assert!(distribution.most_probable_velocity > 0.0);
        assert!(distribution.average_velocity > 0.0);
        assert!(distribution.rms_velocity > 0.0);
        assert!(distribution.rms_velocity > distribution.most_probable_velocity);
    }

    #[test]
    async fn test_vibration_modes() {
        let config = ThermodynamicConfig::default();
        let calculator = KineticTheoryCalculator::new(&config).unwrap();
        
        let modes = calculator.calculate_vibration_modes(300.0, 1.0, &MoleculeType::H2O).await.unwrap();
        assert!(modes.symmetric_stretching > 0.0);
        assert!(modes.asymmetric_stretching > 0.0);
        assert!(modes.bending > 0.0);
        assert!(modes.rotational > 0.0);
        assert!(modes.translational > 0.0);
    }

    #[test]
    async fn test_temperature_from_frequency() {
        let config = ThermodynamicConfig::default();
        let calculator = KineticTheoryCalculator::new(&config).unwrap();
        
        let original_temp = 300.0;
        let frequency = calculator.calculate_oscillation_frequency(original_temp, &MoleculeType::N2).await.unwrap();
        let calculated_temp = calculator.calculate_temperature_from_frequency(frequency.frequency, &MoleculeType::N2).await.unwrap();
        
        assert!((calculated_temp - original_temp).abs() < 1.0); // Within 1K
    }

    #[test]
    async fn test_precision_improvement() {
        let config = ThermodynamicConfig::default();
        let calculator = KineticTheoryCalculator::new(&config).unwrap();
        
        let base_temp = 300.0;
        let temp_increase = 50.0;
        let improvement = calculator.calculate_precision_improvement(temp_increase, base_temp, &MoleculeType::N2).await.unwrap();
        
        assert!(improvement > 1.0); // Should be greater than 1 for temperature increase
    }

    #[test]
    async fn test_effective_temperature() {
        let config = ThermodynamicConfig::default();
        let calculator = KineticTheoryCalculator::new(&config).unwrap();
        
        let mixture = vec![
            (MoleculeType::N2, 0.5),
            (MoleculeType::O2, 0.5),
        ];
        
        let mut temperatures = HashMap::new();
        temperatures.insert(MoleculeType::N2, 300.0);
        temperatures.insert(MoleculeType::O2, 310.0);
        
        let effective_temp = calculator.calculate_effective_temperature(&mixture, &temperatures).await.unwrap();
        assert!(effective_temp > 300.0);
        assert!(effective_temp < 310.0);
    }

    #[test]
    async fn test_thermal_velocity_spread() {
        let config = ThermodynamicConfig::default();
        let calculator = KineticTheoryCalculator::new(&config).unwrap();
        
        let spread = calculator.calculate_thermal_velocity_spread(300.0, &MoleculeType::N2).await.unwrap();
        assert!(spread > 0.0);
    }
} 