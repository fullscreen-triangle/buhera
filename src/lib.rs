//! # Buhera Virtual Processor Architectures
//!
//! A theoretical framework for molecular-scale computational substrates that transcends
//! traditional semiconductor limitations through virtual processor architectures.
//!
//! This crate implements the core components of the Buhera system:
//! - **Masunda Temporal Coordinate Navigator**: Ultra-precise temporal navigation system
//! - Virtual Processing Operating System (VPOS)
//! - Molecular substrate interfaces
//! - Fuzzy digital state management
//! - Quantum coherence management
//! - Biological Maxwell Demon (BMD) information catalysis
//! - Semantic information processing
//! - Neural network integration
//! - Neural pattern transfer protocols
//!
//! ## Architecture Overview
//!
//! The Buhera system operates through a layered architecture:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Application Layer                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │              Semantic Processing Framework                      │
//! ├─────────────────────────────────────────────────────────────────┤
//! │            BMD Information Catalyst Services                    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │             Neural Pattern Transfer Stack                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │              Neural Network Integration                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │              Quantum Coherence Layer                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │            Fuzzy State Management                              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │           Molecular Substrate Interface                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │            Virtual Processor Kernel                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │        Masunda Temporal Coordinate Navigator                   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Getting Started
//!
//! ```rust
//! use buhera::masunda::MasundaNavigator;
//! use buhera::vpos::VirtualProcessorKernel;
//! use buhera::molecular::MolecularSubstrate;
//! use buhera::fuzzy::FuzzyStateManager;
//!
//! // Initialize the Masunda Temporal Coordinate Navigator
//! let navigator = MasundaNavigator::new();
//!
//! // Initialize the VPOS kernel with temporal navigation
//! let kernel = VirtualProcessorKernel::new_with_navigator(navigator);
//!
//! // Create a molecular substrate
//! let substrate = MolecularSubstrate::synthetic_biology_config();
//!
//! // Initialize fuzzy state management
//! let fuzzy_manager = FuzzyStateManager::new();
//!
//! // The system is now ready for temporal-coordinate-precise computation
//! ```
//!
//! ## Key Features
//!
//! - **Masunda Temporal Coordinate Navigator**: 10^-30 second precision timing foundation
//! - **Virtual Processors**: Computational abstractions operating through molecular interactions
//! - **Fuzzy Digital Logic**: Continuous-valued computation transcending binary limitations
//! - **Quantum Coherence**: Biological quantum processing properties
//! - **BMD Information Catalysis**: Entropy reduction through pattern recognition
//! - **Semantic Processing**: Meaning-preserving computational transformations
//! - **Neural Integration**: Biological neural network computational paradigms
//! - **Neural Pattern Transfer**: Direct neural-to-neural information transfer
//!
//! ## Mathematical Foundation
//!
//! The system operates on several key mathematical principles:
//!
//! ### Temporal Coordinate Navigation
//!
//! ```text
//! T_coordinate = T_masunda + Δt_precision × oscillation_convergence
//! ```
//!
//! ### BMD Information Catalysis
//!
//! ```text
//! iCat_comp = I_input ∘ I_output
//! ```
//!
//! ### Fuzzy Digital States
//!
//! ```text
//! Gate_state(t) = f(input_history, process_context, t) ∈ [0,1]
//! ```
//!
//! ### Quantum Coherence
//!
//! ```text
//! τ_coherence = ℏ / (k_B * T_eff)
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(clippy::cargo)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]

/// Masunda Temporal Coordinate Navigator - Ultra-precise temporal navigation system
/// 
/// In memory of Mrs. Stella-Lorraine Masunda
pub mod masunda;

/// Mzekezeke Multi-Dimensional Temporal Ephemeral Cryptography (MDTEC)
/// 
/// Revolutionary cryptographic system using temporal coordinates and recursive precision
pub mod mzekezeke;

/// Virtual Processing Operating System (VPOS) - The core kernel layer
pub mod vpos;

/// Molecular substrate interface and management
pub mod molecular;

/// Fuzzy digital state management and computation
pub mod fuzzy;

/// Quantum coherence management and biological quantum processing
pub mod quantum;

/// Neural network integration and biological computation paradigms
pub mod neural;

/// Neural pattern transfer and direct neural interfaces
pub mod neural_transfer;

/// Biological Maxwell Demon (BMD) information catalysis services
pub mod bmd;

/// Semantic information processing and meaning-preserving computation
pub mod semantic;

/// Molecular foundry system for virtual processor synthesis
pub mod foundry;

/// Mathematical foundations and computational primitives
pub mod math;

/// Error types and result handling
pub mod error;

/// Configuration and system settings
pub mod config;

/// Utilities and helper functions
pub mod utils;

/// Borgia module for integrating BMD systems
pub mod borgia;

/// Zero Computation Engine - Revolutionary direct navigation to predetermined results
pub mod zero_computation;

// Re-export key types for convenience
pub use error::{BuheraError, BuheraResult};
pub use masunda::{
    MasundaNavigator, 
    RecursivePrecisionEngine, 
    VirtualQuantumProcessor,
    TemporalCoordinate,
    InfiniteKey,
    RecursivePrecisionState
};
pub use mzekezeke::{
    MzekezekeSystem,
    MDTECEngine,
    ConsciousnessAwareCrypto,
    TwelveDimensionalSecurity,
    InfiniteKeyGenerator
};
pub use vpos::VirtualProcessorKernel;
pub use molecular::MolecularSubstrate;
pub use fuzzy::FuzzyStateManager;
pub use quantum::QuantumCoherenceLayer;
pub use neural::NeuralIntegration;
pub use neural_transfer::NeuralPatternTransfer;
pub use bmd::BMDCatalyst;
pub use semantic::SemanticProcessor;
pub use foundry::MolecularFoundry;
pub use borgia::{
    IntegratedBMDSystem, BMDScale, InformationCatalyst, MolecularStructure, TurbulanceCompiler,
    CrossScaleAnalysisResult, NavigationStrategy, ConsciousnessAnalysisResult,
    NoiseEnhancedResult, MolecularNavigationResult
};
pub use zero_computation::{
    ZeroComputationEngine,
    ComputationalProblem,
    EntropyEndpoint,
    ZeroComputationMetrics,
};

/// Current version of the Buhera framework
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Framework description
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Initialize the Buhera framework with default configuration
///
/// This function sets up the complete VPOS stack with default parameters
/// suitable for most biological quantum processing tasks.
///
/// # Returns
///
/// A configured `VirtualProcessorKernel` ready for operation
///
/// # Examples
///
/// ```rust
/// use buhera::init_framework;
///
/// let kernel = init_framework();
/// // The system is now ready for biological quantum processing
/// ```
pub fn init_framework() -> BuheraResult<VirtualProcessorKernel> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Create the kernel with default configuration
    let kernel = VirtualProcessorKernel::new()?;
    
    tracing::info!("Buhera Virtual Processor Architectures framework initialized");
    tracing::info!("Version: {}", VERSION);
    tracing::info!("Ready for biological quantum processing");
    
    Ok(kernel)
}

/// Initialize the framework with custom configuration
///
/// This function allows for advanced configuration of the VPOS system
/// for specialized biological quantum processing tasks.
///
/// # Arguments
///
/// * `config` - Custom configuration parameters
///
/// # Returns
///
/// A configured `VirtualProcessorKernel` with custom settings
///
/// # Examples
///
/// ```rust
/// use buhera::{init_framework_with_config, config::BuheraConfig};
///
/// let config = BuheraConfig::builder()
///     .quantum_coherence_enabled(true)
///     .fuzzy_logic_precision(0.001)
///     .molecular_substrate_type("synthetic_biology")
///     .neural_pattern_transfer_enabled(true)
///     .build();
///
/// let kernel = init_framework_with_config(config)?;
/// ```
pub fn init_framework_with_config(config: config::BuheraConfig) -> BuheraResult<VirtualProcessorKernel> {
    // Initialize logging with custom configuration
    tracing_subscriber::fmt::init();
    
    // Create the kernel with custom configuration
    let kernel = VirtualProcessorKernel::with_config(config)?;
    
    tracing::info!("Buhera framework initialized with custom configuration");
    tracing::info!("Version: {}", VERSION);
    tracing::info!("Configuration: {:?}", kernel.config());
    
    Ok(kernel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_initialization() {
        let result = init_framework();
        assert!(result.is_ok());
    }

    #[test]
    fn test_version_defined() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_description_defined() {
        assert!(!DESCRIPTION.is_empty());
        assert!(DESCRIPTION.contains("Buhera"));
    }
} 