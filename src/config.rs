//! Configuration management for the Buhera Virtual Processor Architectures framework

use serde::{Deserialize, Serialize};
use std::path::Path;
use crate::error::{BuheraResult, ConfigError};

/// Main configuration structure for the Buhera framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuheraConfig {
    /// VPOS kernel configuration
    pub vpos: VposConfig,
    
    /// Molecular substrate configuration
    pub molecular: MolecularConfig,
    
    /// Fuzzy logic configuration
    pub fuzzy: FuzzyConfig,
    
    /// Quantum coherence configuration
    pub quantum: QuantumConfig,
    
    /// Neural integration configuration
    pub neural: NeuralConfig,
    
    /// Neural pattern transfer configuration
    pub neural_transfer: NeuralTransferConfig,
    
    /// BMD catalyst configuration
    pub bmd: BmdConfig,
    
    /// Semantic processing configuration
    pub semantic: SemanticConfig,
    
    /// Molecular foundry configuration
    pub foundry: FoundryConfig,
    
    /// System configuration
    pub system: SystemConfig,
}

/// VPOS kernel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VposConfig {
    /// Enable daemon mode
    pub daemon_mode: bool,
    
    /// Bind address for kernel services
    pub bind_address: String,
    
    /// Maximum number of virtual processors
    pub max_virtual_processors: u32,
    
    /// Scheduler algorithm
    pub scheduler_algorithm: String,
    
    /// Process timeout in seconds
    pub process_timeout: u64,
    
    /// Resource allocation strategy
    pub resource_allocation: String,
}

/// Molecular substrate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularConfig {
    /// Substrate type
    pub substrate_type: String,
    
    /// Temperature in Celsius
    pub temperature: f64,
    
    /// pH level
    pub ph: f64,
    
    /// Ionic strength in mM
    pub ionic_strength: f64,
    
    /// Protein synthesis rate
    pub protein_synthesis_rate: f64,
    
    /// Enable environmental monitoring
    pub environmental_monitoring: bool,
}

/// Fuzzy logic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyConfig {
    /// Fuzzy precision
    pub precision: f64,
    
    /// Defuzzification method
    pub defuzzification_method: String,
    
    /// Enable fuzzy memory
    pub fuzzy_memory: bool,
    
    /// Membership function type
    pub membership_function: String,
    
    /// Fuzzy rule engine
    pub rule_engine: String,
}

/// Quantum coherence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    /// Enable quantum coherence
    pub enabled: bool,
    
    /// Coherence time in microseconds
    pub coherence_time: f64,
    
    /// Maximum qubits
    pub max_qubits: u32,
    
    /// Decoherence threshold
    pub decoherence_threshold: f64,
    
    /// Quantum error correction
    pub error_correction: bool,
}

/// Neural integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    /// Enable neural integration
    pub enabled: bool,
    
    /// Neural architecture
    pub architecture: String,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Plasticity enabled
    pub plasticity_enabled: bool,
    
    /// Synaptic timeout
    pub synaptic_timeout: u64,
}

/// Neural pattern transfer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralTransferConfig {
    /// Enable neural pattern transfer
    pub enabled: bool,
    
    /// Transfer protocol
    pub protocol: String,
    
    /// Signal strength threshold
    pub signal_threshold: f64,
    
    /// Membrane quantum tunneling enabled
    pub quantum_tunneling: bool,
    
    /// Pattern extraction method
    pub pattern_extraction_method: String,
}

/// BMD catalyst configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmdConfig {
    /// Pattern recognition threshold
    pub pattern_threshold: f64,
    
    /// Information catalysis method
    pub catalysis_method: String,
    
    /// Entropy reduction target
    pub entropy_reduction_target: f64,
    
    /// Filter sensitivity
    pub filter_sensitivity: f64,
    
    /// Channel capacity
    pub channel_capacity: f64,
}

/// Semantic processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    /// Semantic coherence threshold
    pub coherence_threshold: f64,
    
    /// Cross-modal processing enabled
    pub cross_modal_enabled: bool,
    
    /// Meaning preservation method
    pub meaning_preservation_method: String,
    
    /// Semantic verification enabled
    pub semantic_verification: bool,
    
    /// Context processing depth
    pub context_processing_depth: u32,
}

/// Molecular foundry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoundryConfig {
    /// Number of synthesis chambers
    pub synthesis_chambers: u32,
    
    /// Quality control level
    pub quality_control_level: String,
    
    /// Assembly automation enabled
    pub assembly_automation: bool,
    
    /// Synthesis protocol
    pub synthesis_protocol: String,
    
    /// Resource allocation strategy
    pub resource_allocation: String,
}

/// System configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Logging level
    pub log_level: String,
    
    /// Enable metrics collection
    pub metrics_enabled: bool,
    
    /// Data directory
    pub data_directory: String,
    
    /// Configuration file path
    pub config_file: String,
    
    /// Enable debug mode
    pub debug_mode: bool,
}

impl Default for BuheraConfig {
    fn default() -> Self {
        Self {
            vpos: VposConfig::default(),
            molecular: MolecularConfig::default(),
            fuzzy: FuzzyConfig::default(),
            quantum: QuantumConfig::default(),
            neural: NeuralConfig::default(),
            neural_transfer: NeuralTransferConfig::default(),
            bmd: BmdConfig::default(),
            semantic: SemanticConfig::default(),
            foundry: FoundryConfig::default(),
            system: SystemConfig::default(),
        }
    }
}

impl Default for VposConfig {
    fn default() -> Self {
        Self {
            daemon_mode: false,
            bind_address: "127.0.0.1:8080".to_string(),
            max_virtual_processors: 1000,
            scheduler_algorithm: "fuzzy_round_robin".to_string(),
            process_timeout: 3600,
            resource_allocation: "adaptive".to_string(),
        }
    }
}

impl Default for MolecularConfig {
    fn default() -> Self {
        Self {
            substrate_type: "synthetic_biology".to_string(),
            temperature: 37.0,
            ph: 7.4,
            ionic_strength: 150.0,
            protein_synthesis_rate: 1.0,
            environmental_monitoring: true,
        }
    }
}

impl Default for FuzzyConfig {
    fn default() -> Self {
        Self {
            precision: 0.001,
            defuzzification_method: "centroid".to_string(),
            fuzzy_memory: true,
            membership_function: "triangular".to_string(),
            rule_engine: "mamdani".to_string(),
        }
    }
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            coherence_time: 100.0,
            max_qubits: 32,
            decoherence_threshold: 0.1,
            error_correction: true,
        }
    }
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            architecture: "multilayer_perceptron".to_string(),
            learning_rate: 0.001,
            plasticity_enabled: true,
            synaptic_timeout: 1000,
        }
    }
}

impl Default for NeuralTransferConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            protocol: "membrane_quantum_interface".to_string(),
            signal_threshold: 0.8,
            quantum_tunneling: true,
            pattern_extraction_method: "ion_channel_pattern_matching".to_string(),
        }
    }
}

impl Default for BmdConfig {
    fn default() -> Self {
        Self {
            pattern_threshold: 0.8,
            catalysis_method: "entropy_reduction".to_string(),
            entropy_reduction_target: 0.5,
            filter_sensitivity: 0.9,
            channel_capacity: 1000.0,
        }
    }
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            coherence_threshold: 0.7,
            cross_modal_enabled: true,
            meaning_preservation_method: "semantic_coherence".to_string(),
            semantic_verification: true,
            context_processing_depth: 5,
        }
    }
}

impl Default for FoundryConfig {
    fn default() -> Self {
        Self {
            synthesis_chambers: 4,
            quality_control_level: "high".to_string(),
            assembly_automation: true,
            synthesis_protocol: "standard".to_string(),
            resource_allocation: "balanced".to_string(),
        }
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            metrics_enabled: true,
            data_directory: "data".to_string(),
            config_file: "buhera.toml".to_string(),
            debug_mode: false,
        }
    }
}

impl BuheraConfig {
    /// Create a new configuration builder
    pub fn builder() -> BuheraConfigBuilder {
        BuheraConfigBuilder::new()
    }
    
    /// Load configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> BuheraResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ConfigError::ConfigurationFileError {
                file: path.as_ref().to_string_lossy().to_string(),
            })?;
        
        toml::from_str(&content)
            .map_err(|_| ConfigError::ConfigurationParsingError {
                format: "TOML".to_string(),
            })
    }
    
    /// Save configuration to file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> BuheraResult<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|_| ConfigError::ConfigurationParsingError {
                format: "TOML".to_string(),
            })?;
        
        std::fs::write(path.as_ref(), content)
            .map_err(|e| ConfigError::ConfigurationFileError {
                file: path.as_ref().to_string_lossy().to_string(),
            })?;
        
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> BuheraResult<()> {
        // Validate molecular configuration
        if self.molecular.temperature < 0.0 || self.molecular.temperature > 100.0 {
            return Err(ConfigError::InvalidValue {
                key: "molecular.temperature".to_string(),
                value: self.molecular.temperature.to_string(),
            }.into());
        }
        
        if self.molecular.ph < 0.0 || self.molecular.ph > 14.0 {
            return Err(ConfigError::InvalidValue {
                key: "molecular.ph".to_string(),
                value: self.molecular.ph.to_string(),
            }.into());
        }
        
        // Validate fuzzy configuration
        if self.fuzzy.precision <= 0.0 || self.fuzzy.precision > 1.0 {
            return Err(ConfigError::InvalidValue {
                key: "fuzzy.precision".to_string(),
                value: self.fuzzy.precision.to_string(),
            }.into());
        }
        
        // Validate quantum configuration
        if self.quantum.coherence_time <= 0.0 {
            return Err(ConfigError::InvalidValue {
                key: "quantum.coherence_time".to_string(),
                value: self.quantum.coherence_time.to_string(),
            }.into());
        }
        
        Ok(())
    }
}

/// Configuration builder for fluent API
pub struct BuheraConfigBuilder {
    config: BuheraConfig,
}

impl BuheraConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: BuheraConfig::default(),
        }
    }
    
    /// Set quantum coherence enabled
    pub fn quantum_coherence_enabled(mut self, enabled: bool) -> Self {
        self.config.quantum.enabled = enabled;
        self
    }
    
    /// Set fuzzy logic precision
    pub fn fuzzy_logic_precision(mut self, precision: f64) -> Self {
        self.config.fuzzy.precision = precision;
        self
    }
    
    /// Set molecular substrate type
    pub fn molecular_substrate_type(mut self, substrate_type: &str) -> Self {
        self.config.molecular.substrate_type = substrate_type.to_string();
        self
    }
    
    /// Set neural integration enabled
    pub fn neural_integration_enabled(mut self, enabled: bool) -> Self {
        self.config.neural.enabled = enabled;
        self
    }
    
    /// Set neural pattern transfer enabled
    pub fn neural_pattern_transfer_enabled(mut self, enabled: bool) -> Self {
        self.config.neural_transfer.enabled = enabled;
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> BuheraConfig {
        self.config
    }
}

impl Default for BuheraConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = BuheraConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = BuheraConfig::builder()
            .quantum_coherence_enabled(true)
            .fuzzy_logic_precision(0.001)
            .molecular_substrate_type("synthetic_biology")
            .build();
        
        assert!(config.quantum.enabled);
        assert_eq!(config.fuzzy.precision, 0.001);
        assert_eq!(config.molecular.substrate_type, "synthetic_biology");
    }

    #[test]
    fn test_config_serialization() {
        let config = BuheraConfig::default();
        let serialized = toml::to_string(&config).unwrap();
        let deserialized: BuheraConfig = toml::from_str(&serialized).unwrap();
        
        assert_eq!(config.vpos.bind_address, deserialized.vpos.bind_address);
    }

    #[test]
    fn test_config_file_operations() {
        let config = BuheraConfig::default();
        let temp_file = NamedTempFile::new().unwrap();
        
        config.to_file(temp_file.path()).unwrap();
        let loaded_config = BuheraConfig::from_file(temp_file.path()).unwrap();
        
        assert_eq!(config.vpos.bind_address, loaded_config.vpos.bind_address);
    }

    #[test]
    fn test_config_validation() {
        let mut config = BuheraConfig::default();
        
        // Valid config should pass
        assert!(config.validate().is_ok());
        
        // Invalid temperature should fail
        config.molecular.temperature = -10.0;
        assert!(config.validate().is_err());
        
        // Invalid pH should fail
        config.molecular.temperature = 37.0;
        config.molecular.ph = 20.0;
        assert!(config.validate().is_err());
    }
} 