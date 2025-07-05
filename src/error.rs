//! Error handling for the Buhera Virtual Processor Architectures framework
//!
//! This module defines all error types that can occur within the Buhera system,
//! from molecular substrate failures to quantum coherence losses.

use std::fmt;
use thiserror::Error;

/// Result type for Buhera operations
pub type BuheraResult<T> = Result<T, BuheraError>;

/// Main error type for the Buhera framework
#[derive(Error, Debug)]
pub enum BuheraError {
    /// Virtual processor kernel errors
    #[error("VPOS kernel error: {0}")]
    VposKernel(#[from] VposError),

    /// Molecular substrate interface errors
    #[error("Molecular substrate error: {0}")]
    MolecularSubstrate(#[from] MolecularError),

    /// Fuzzy state management errors
    #[error("Fuzzy state error: {0}")]
    FuzzyState(#[from] FuzzyError),

    /// Quantum coherence errors
    #[error("Quantum coherence error: {0}")]
    QuantumCoherence(#[from] QuantumError),

    /// Neural integration errors
    #[error("Neural integration error: {0}")]
    NeuralIntegration(#[from] NeuralError),

    /// Neural pattern transfer errors
    #[error("Neural pattern transfer error: {0}")]
    NeuralPatternTransfer(#[from] NeuralTransferError),

    /// BMD information catalysis errors
    #[error("BMD catalyst error: {0}")]
    BmdCatalyst(#[from] BmdError),

    /// Semantic processing errors
    #[error("Semantic processing error: {0}")]
    SemanticProcessing(#[from] SemanticError),

    /// Molecular foundry errors
    #[error("Molecular foundry error: {0}")]
    MolecularFoundry(#[from] FoundryError),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(#[from] ConfigError),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Network errors
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Generic error with message
    #[error("Generic error: {0}")]
    Generic(String),
}

/// VPOS kernel specific errors
#[derive(Error, Debug)]
pub enum VposError {
    /// Kernel initialization failed
    #[error("Failed to initialize VPOS kernel: {reason}")]
    InitializationFailed { reason: String },

    /// Virtual processor creation failed
    #[error("Failed to create virtual processor: {processor_type}")]
    ProcessorCreationFailed { processor_type: String },

    /// Scheduling error
    #[error("Scheduling error: {reason}")]
    SchedulingError { reason: String },

    /// Process state transition error
    #[error("Invalid process state transition: {from} -> {to}")]
    InvalidStateTransition { from: String, to: String },

    /// Resource allocation error
    #[error("Resource allocation failed: {resource}")]
    ResourceAllocation { resource: String },
}

/// Molecular substrate errors
#[derive(Error, Debug)]
pub enum MolecularError {
    /// Protein synthesis failed
    #[error("Protein synthesis failed: {protein_type}")]
    ProteinSynthesisFailed { protein_type: String },

    /// Conformational change error
    #[error("Conformational change failed: {protein_id}")]
    ConformationalChangeFailed { protein_id: String },

    /// Enzymatic reaction error
    #[error("Enzymatic reaction failed: {enzyme}, {substrate}")]
    EnzymaticReactionFailed { enzyme: String, substrate: String },

    /// Molecular assembly error
    #[error("Molecular assembly failed: {component}")]
    MolecularAssemblyFailed { component: String },

    /// Substrate degradation
    #[error("Substrate degradation detected: {substrate_id}")]
    SubstrateDegradation { substrate_id: String },

    /// Environmental conditions error
    #[error("Environmental conditions out of range: {parameter} = {value}")]
    EnvironmentalConditions { parameter: String, value: f64 },
}

/// Fuzzy state management errors
#[derive(Error, Debug)]
pub enum FuzzyError {
    /// Invalid fuzzy value
    #[error("Invalid fuzzy value: {value} (must be in range [0,1])")]
    InvalidFuzzyValue { value: f64 },

    /// Fuzzy operation error
    #[error("Fuzzy operation failed: {operation}")]
    FuzzyOperationFailed { operation: String },

    /// Defuzzification error
    #[error("Defuzzification failed: {method}")]
    DefuzzificationFailed { method: String },

    /// Membership function error
    #[error("Membership function error: {function}")]
    MembershipFunctionError { function: String },

    /// Fuzzy rule error
    #[error("Fuzzy rule error: {rule}")]
    FuzzyRuleError { rule: String },
}

/// Quantum coherence errors
#[derive(Error, Debug)]
pub enum QuantumError {
    /// Quantum coherence loss
    #[error("Quantum coherence lost: {coherence_time}μs")]
    CoherenceLoss { coherence_time: f64 },

    /// Quantum decoherence
    #[error("Quantum decoherence detected: {decoherence_rate}")]
    QuantumDecoherence { decoherence_rate: f64 },

    /// Quantum state preparation error
    #[error("Quantum state preparation failed: {state}")]
    StatePreparationFailed { state: String },

    /// Quantum measurement error
    #[error("Quantum measurement error: {qubit_id}")]
    MeasurementError { qubit_id: u32 },

    /// Quantum gate error
    #[error("Quantum gate operation failed: {gate_type}")]
    QuantumGateError { gate_type: String },

    /// Entanglement error
    #[error("Entanglement operation failed: {qubits:?}")]
    EntanglementError { qubits: Vec<u32> },
}

/// Neural integration errors
#[derive(Error, Debug)]
pub enum NeuralError {
    /// Neural network initialization error
    #[error("Neural network initialization failed: {architecture}")]
    NetworkInitializationFailed { architecture: String },

    /// Training error
    #[error("Neural network training failed: {epoch}")]
    TrainingFailed { epoch: u32 },

    /// Inference error
    #[error("Neural network inference failed: {input_shape:?}")]
    InferenceFailed { input_shape: Vec<usize> },

    /// Synaptic update error
    #[error("Synaptic update failed: {synapse_id}")]
    SynapticUpdateFailed { synapse_id: String },

    /// Plasticity error
    #[error("Plasticity mechanism failed: {mechanism}")]
    PlasticityError { mechanism: String },

    /// Neural communication error
    #[error("Neural communication failed: {source} -> {target}")]
    NeuralCommunicationError { source: String, target: String },
}

/// Neural pattern transfer errors
#[derive(Error, Debug)]
pub enum NeuralTransferError {
    /// Pattern extraction failed
    #[error("Pattern extraction failed: {pattern_type}")]
    PatternExtractionFailed { pattern_type: String },

    /// Pattern transfer failed
    #[error("Pattern transfer failed: {target_id}")]
    PatternTransferFailed { target_id: String },

    /// Neural interface error
    #[error("Neural interface error: {interface_type}")]
    NeuralInterfaceError { interface_type: String },

    /// Transfer protocol error
    #[error("Transfer protocol error: {protocol}")]
    TransferProtocolError { protocol: String },

    /// Signal degradation
    #[error("Signal degradation detected: {signal_strength}")]
    SignalDegradation { signal_strength: f64 },

    /// Membrane quantum tunneling error
    #[error("Membrane quantum tunneling error: {channel_id}")]
    QuantumTunnelingError { channel_id: String },
}

/// BMD information catalysis errors
#[derive(Error, Debug)]
pub enum BmdError {
    /// Pattern recognition failed
    #[error("Pattern recognition failed: {pattern_type}")]
    PatternRecognitionFailed { pattern_type: String },

    /// Information catalysis failed
    #[error("Information catalysis failed: {catalyst_type}")]
    InformationCatalysisFailed { catalyst_type: String },

    /// Entropy reduction failed
    #[error("Entropy reduction failed: {expected_reduction}")]
    EntropyReductionFailed { expected_reduction: f64 },

    /// Filter operation failed
    #[error("Filter operation failed: {filter_type}")]
    FilterOperationFailed { filter_type: String },

    /// Channel operation failed
    #[error("Channel operation failed: {channel_type}")]
    ChannelOperationFailed { channel_type: String },

    /// Information ordering failed
    #[error("Information ordering failed: {order_type}")]
    InformationOrderingFailed { order_type: String },
}

/// Semantic processing errors
#[derive(Error, Debug)]
pub enum SemanticError {
    /// Semantic encoding failed
    #[error("Semantic encoding failed: {content_type}")]
    SemanticEncodingFailed { content_type: String },

    /// Semantic decoding failed
    #[error("Semantic decoding failed: {target_type}")]
    SemanticDecodingFailed { target_type: String },

    /// Meaning preservation failed
    #[error("Meaning preservation failed: {semantic_coherence}")]
    MeaningPreservationFailed { semantic_coherence: f64 },

    /// Cross-modal transformation failed
    #[error("Cross-modal transformation failed: {from} -> {to}")]
    CrossModalTransformationFailed { from: String, to: String },

    /// Semantic context error
    #[error("Semantic context error: {context}")]
    SemanticContextError { context: String },

    /// Semantic verification failed
    #[error("Semantic verification failed: {verification_type}")]
    SemanticVerificationFailed { verification_type: String },
}

/// Molecular foundry errors
#[derive(Error, Debug)]
pub enum FoundryError {
    /// Synthesis chamber error
    #[error("Synthesis chamber error: {chamber_id}")]
    SynthesisChamberError { chamber_id: String },

    /// Template error
    #[error("Template error: {template_type}")]
    TemplateError { template_type: String },

    /// Quality control failed
    #[error("Quality control failed: {test_type}")]
    QualityControlFailed { test_type: String },

    /// Assembly automation error
    #[error("Assembly automation error: {component}")]
    AssemblyAutomationError { component: String },

    /// Synthesis protocol error
    #[error("Synthesis protocol error: {protocol}")]
    SynthesisProtocolError { protocol: String },

    /// Foundry resource error
    #[error("Foundry resource error: {resource}")]
    FoundryResourceError { resource: String },
}

/// Configuration errors
#[derive(Error, Debug)]
pub enum ConfigError {
    /// Invalid configuration value
    #[error("Invalid configuration value: {key} = {value}")]
    InvalidValue { key: String, value: String },

    /// Missing configuration
    #[error("Missing configuration: {key}")]
    MissingConfiguration { key: String },

    /// Configuration file error
    #[error("Configuration file error: {file}")]
    ConfigurationFileError { file: String },

    /// Environment variable error
    #[error("Environment variable error: {variable}")]
    EnvironmentVariableError { variable: String },

    /// Configuration parsing error
    #[error("Configuration parsing error: {format}")]
    ConfigurationParsingError { format: String },
}

impl BuheraError {
    /// Create a generic error with a message
    pub fn generic<S: Into<String>>(message: S) -> Self {
        Self::Generic(message.into())
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::VposKernel(VposError::SchedulingError { .. }) => true,
            Self::MolecularSubstrate(MolecularError::EnvironmentalConditions { .. }) => true,
            Self::QuantumCoherence(QuantumError::CoherenceLoss { .. }) => true,
                         Self::NeuralPatternTransfer(NeuralTransferError::SignalDegradation { .. }) => true,
            Self::Network(_) => true,
            _ => false,
        }
    }

    /// Get the error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::VposKernel(VposError::InitializationFailed { .. }) => ErrorSeverity::Critical,
            Self::MolecularSubstrate(MolecularError::SubstrateDegradation { .. }) => ErrorSeverity::Critical,
            Self::QuantumCoherence(QuantumError::QuantumDecoherence { .. }) => ErrorSeverity::High,
                         Self::NeuralPatternTransfer(NeuralTransferError::PatternTransferFailed { .. }) => ErrorSeverity::High,
            Self::Configuration(_) => ErrorSeverity::Medium,
            Self::Network(_) => ErrorSeverity::Low,
            _ => ErrorSeverity::Medium,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Critical system errors that require immediate attention
    Critical,
    /// High priority errors that may cause system instability
    High,
    /// Medium priority errors that may affect functionality
    Medium,
    /// Low priority errors that are mainly informational
    Low,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "CRITICAL"),
            Self::High => write!(f, "HIGH"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::Low => write!(f, "LOW"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_severity() {
        let error = BuheraError::VposKernel(VposError::InitializationFailed {
            reason: "test".to_string(),
        });
        assert_eq!(error.severity(), ErrorSeverity::Critical);
    }

    #[test]
    fn test_recoverable_error() {
        let error = BuheraError::VposKernel(VposError::SchedulingError {
            reason: "test".to_string(),
        });
        assert!(error.is_recoverable());
    }

    #[test]
    fn test_generic_error() {
        let error = BuheraError::generic("Test error message");
        assert!(matches!(error, BuheraError::Generic(_)));
    }
} 