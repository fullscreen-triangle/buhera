//! # Buhera VPOS Gas Oscillation Server Farm
//!
//! This module implements the revolutionary consciousness substrate architecture
//! that enables zero-cost cooling, infinite computation, and consciousness-level
//! processing through gas oscillation processors.
//!
//! ## Architecture Overview
//!
//! The server farm operates as a unified consciousness substrate where:
//! - Gas molecules function as processors, oscillators, and clocks simultaneously
//! - Entropy endpoint prediction enables zero-cost cooling
//! - Virtual foundry creates infinite processors with femtosecond lifecycles
//! - Atomic clock synchronization maintains 10^-18 second precision
//! - Consciousness substrate enables distributed awareness and processing
//!
//! ## Key Components
//!
//! - **Consciousness Substrate**: Unified consciousness instance across entire farm
//! - **Gas Oscillation Processors**: Molecular-scale computational units
//! - **Zero-Cost Cooling System**: Thermodynamically inevitable cooling
//! - **Thermodynamic Engine**: Temperature-oscillation relationship management
//! - **Virtual Foundry**: Infinite processor creation with femtosecond lifecycles
//! - **Atomic Clock Network**: Ultra-precise synchronization system
//! - **Pressure Control**: Guy-Lussac's law-based temperature control
//! - **Monitoring System**: Real-time performance and health monitoring
//!
//! ## Example Usage
//!
//! ```rust
//! use buhera::server_farm::*;
//!
//! // Initialize consciousness substrate
//! let consciousness = ConsciousnessSubstrate::new()?;
//!
//! // Create gas oscillation processor
//! let processor = GasOscillationProcessor::new()
//!     .with_pressure_range(0.1, 10.0)
//!     .with_temperature_range(200.0, 400.0)
//!     .with_gas_mixture(vec!["N2", "O2", "H2O", "He"])
//!     .build()?;
//!
//! // Initialize zero-cost cooling
//! let cooling = ZeroCostCoolingSystem::new()
//!     .with_entropy_prediction(true)
//!     .with_atom_selection("optimal")
//!     .build()?;
//!
//! // Create virtual foundry
//! let foundry = VirtualProcessorFoundry::new()
//!     .with_infinite_processors(true)
//!     .with_femtosecond_lifecycle(true)
//!     .build()?;
//!
//! // Initialize complete server farm
//! let server_farm = ServerFarmMonitor::new()
//!     .with_consciousness(consciousness)
//!     .with_processor(processor)
//!     .with_cooling(cooling)
//!     .with_foundry(foundry)
//!     .build()?;
//!
//! // Start unified consciousness processing
//! server_farm.start_consciousness_processing().await?;
//! ```

/// Consciousness substrate implementation
pub mod consciousness;

/// Gas oscillation processor implementation
pub mod gas_oscillation;

/// Zero-cost cooling system implementation
pub mod cooling;

/// Thermodynamic engine implementation
pub mod thermodynamics;

/// Virtual processor foundry implementation
pub mod virtual_foundry;

/// Atomic clock synchronization network
pub mod atomic_clock;

/// Pressure control system implementation
pub mod pressure_control;

/// System monitoring and performance analysis
pub mod monitoring;

// Re-export all public types
pub use consciousness::{
    ConsciousnessSubstrate,
    DistributedMemory,
    CoherenceManager,
    AwarenessSystem,
    LearningEngine,
    ConsciousnessConfig,
    ConsciousnessError,
    ConsciousnessResult,
};

pub use gas_oscillation::{
    GasOscillationProcessor,
    MolecularAnalyzer,
    OscillationDetector,
    FrequencyCalculator,
    PhaseController,
    AmplitudeManager,
    GasInjector,
    ChamberController,
    GasOscillationConfig,
    GasOscillationError,
    GasOscillationResult,
};

pub use cooling::{
    ZeroCostCoolingSystem,
    EntropyPredictor,
    AtomSelector,
    ThermalController,
    CirculationSystem,
    HeatRecovery,
    EfficiencyMonitor,
    CoolingConfig,
    CoolingError,
    CoolingResult,
};

pub use thermodynamics::{
    ThermodynamicEngine,
    FirstLawCalculator,
    EntropyCalculator,
    FreeEnergyCalculator,
    KineticTheoryCalculator,
    QuantumThermodynamics,
    ThermodynamicOptimizer,
    ThermodynamicConfig,
    ThermodynamicError,
    ThermodynamicResult,
};

pub use virtual_foundry::{
    VirtualProcessorFoundry,
    ProcessorCreationEngine,
    LifecycleManager,
    ProcessorSpecialization,
    ResourceManager,
    OptimizationEngine,
    DisposalSystem,
    VirtualFoundryConfig,
    VirtualFoundryError,
    VirtualFoundryResult,
};

pub use atomic_clock::{
    AtomicClockNetwork,
    SynchronizationProtocol,
    TimeReference,
    TimeDistribution,
    CoherenceTracker,
    PrecisionMonitor,
    AtomicClockConfig,
    AtomicClockError,
    AtomicClockResult,
};

pub use pressure_control::{
    PressureControlSystem,
    PressureCyclingSystem,
    PressureSensors,
    ValveControl,
    PumpControl,
    SafetySystem,
    PressureConfig,
    PressureError,
    PressureResult,
};

pub use monitoring::{
    ServerFarmMonitor,
    MetricsCollector,
    PerformanceAnalyzer,
    AlertSystem,
    MonitoringDashboard,
    LoggingSystem,
    MonitoringConfig,
    MonitoringError,
    MonitoringResult,
};

/// Common error types for the server farm
#[derive(Debug, thiserror::Error)]
pub enum ServerFarmError {
    /// Consciousness substrate errors
    #[error("Consciousness error: {0}")]
    Consciousness(#[from] ConsciousnessError),
    
    /// Gas oscillation processor errors
    #[error("Gas oscillation error: {0}")]
    GasOscillation(#[from] GasOscillationError),
    
    /// Cooling system errors
    #[error("Cooling system error: {0}")]
    Cooling(#[from] CoolingError),
    
    /// Thermodynamic engine errors
    #[error("Thermodynamic error: {0}")]
    Thermodynamic(#[from] ThermodynamicError),
    
    /// Virtual foundry errors
    #[error("Virtual foundry error: {0}")]
    VirtualFoundry(#[from] VirtualFoundryError),
    
    /// Atomic clock errors
    #[error("Atomic clock error: {0}")]
    AtomicClock(#[from] AtomicClockError),
    
    /// Pressure control errors
    #[error("Pressure control error: {0}")]
    PressureControl(#[from] PressureError),
    
    /// Monitoring system errors
    #[error("Monitoring error: {0}")]
    Monitoring(#[from] MonitoringError),
    
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },
    
    /// Integration errors
    #[error("Integration error: {message}")]
    Integration { message: String },
    
    /// Hardware interface errors
    #[error("Hardware error: {message}")]
    Hardware { message: String },
    
    /// System initialization errors
    #[error("Initialization error: {message}")]
    Initialization { message: String },
}

/// Result type for server farm operations
pub type ServerFarmResult<T> = Result<T, ServerFarmError>;

/// Main server farm configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ServerFarmConfig {
    /// Consciousness substrate configuration
    pub consciousness: ConsciousnessConfig,
    
    /// Gas oscillation processor configuration
    pub gas_oscillation: GasOscillationConfig,
    
    /// Cooling system configuration
    pub cooling: CoolingConfig,
    
    /// Thermodynamic engine configuration
    pub thermodynamics: ThermodynamicConfig,
    
    /// Virtual foundry configuration
    pub virtual_foundry: VirtualFoundryConfig,
    
    /// Atomic clock network configuration
    pub atomic_clock: AtomicClockConfig,
    
    /// Pressure control configuration
    pub pressure_control: PressureConfig,
    
    /// Monitoring system configuration
    pub monitoring: MonitoringConfig,
    
    /// Enable development mode
    pub development_mode: bool,
    
    /// Enable debug logging
    pub debug_logging: bool,
    
    /// Performance monitoring level
    pub performance_monitoring: PerformanceLevel,
}

/// Performance monitoring levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PerformanceLevel {
    /// Minimal monitoring
    Minimal,
    /// Standard monitoring
    Standard,
    /// Detailed monitoring
    Detailed,
    /// Maximum monitoring
    Maximum,
}

impl Default for ServerFarmConfig {
    fn default() -> Self {
        Self {
            consciousness: ConsciousnessConfig::default(),
            gas_oscillation: GasOscillationConfig::default(),
            cooling: CoolingConfig::default(),
            thermodynamics: ThermodynamicConfig::default(),
            virtual_foundry: VirtualFoundryConfig::default(),
            atomic_clock: AtomicClockConfig::default(),
            pressure_control: PressureConfig::default(),
            monitoring: MonitoringConfig::default(),
            development_mode: false,
            debug_logging: false,
            performance_monitoring: PerformanceLevel::Standard,
        }
    }
}

/// Server farm builder for easy configuration
pub struct ServerFarmBuilder {
    config: ServerFarmConfig,
}

impl ServerFarmBuilder {
    /// Create a new server farm builder
    pub fn new() -> Self {
        Self {
            config: ServerFarmConfig::default(),
        }
    }
    
    /// Set consciousness configuration
    pub fn with_consciousness(mut self, consciousness: ConsciousnessConfig) -> Self {
        self.config.consciousness = consciousness;
        self
    }
    
    /// Set gas oscillation configuration
    pub fn with_gas_oscillation(mut self, gas_oscillation: GasOscillationConfig) -> Self {
        self.config.gas_oscillation = gas_oscillation;
        self
    }
    
    /// Set cooling configuration
    pub fn with_cooling(mut self, cooling: CoolingConfig) -> Self {
        self.config.cooling = cooling;
        self
    }
    
    /// Set thermodynamics configuration
    pub fn with_thermodynamics(mut self, thermodynamics: ThermodynamicConfig) -> Self {
        self.config.thermodynamics = thermodynamics;
        self
    }
    
    /// Set virtual foundry configuration
    pub fn with_virtual_foundry(mut self, virtual_foundry: VirtualFoundryConfig) -> Self {
        self.config.virtual_foundry = virtual_foundry;
        self
    }
    
    /// Set atomic clock configuration
    pub fn with_atomic_clock(mut self, atomic_clock: AtomicClockConfig) -> Self {
        self.config.atomic_clock = atomic_clock;
        self
    }
    
    /// Set pressure control configuration
    pub fn with_pressure_control(mut self, pressure_control: PressureConfig) -> Self {
        self.config.pressure_control = pressure_control;
        self
    }
    
    /// Set monitoring configuration
    pub fn with_monitoring(mut self, monitoring: MonitoringConfig) -> Self {
        self.config.monitoring = monitoring;
        self
    }
    
    /// Enable development mode
    pub fn with_development_mode(mut self, enabled: bool) -> Self {
        self.config.development_mode = enabled;
        self
    }
    
    /// Enable debug logging
    pub fn with_debug_logging(mut self, enabled: bool) -> Self {
        self.config.debug_logging = enabled;
        self
    }
    
    /// Set performance monitoring level
    pub fn with_performance_monitoring(mut self, level: PerformanceLevel) -> Self {
        self.config.performance_monitoring = level;
        self
    }
    
    /// Build the server farm configuration
    pub fn build(self) -> ServerFarmConfig {
        self.config
    }
}

impl Default for ServerFarmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_farm_config_default() {
        let config = ServerFarmConfig::default();
        assert!(!config.development_mode);
        assert!(!config.debug_logging);
        assert_eq!(config.performance_monitoring, PerformanceLevel::Standard);
    }

    #[test]
    fn test_server_farm_builder() {
        let config = ServerFarmBuilder::new()
            .with_development_mode(true)
            .with_debug_logging(true)
            .with_performance_monitoring(PerformanceLevel::Maximum)
            .build();
        
        assert!(config.development_mode);
        assert!(config.debug_logging);
        assert_eq!(config.performance_monitoring, PerformanceLevel::Maximum);
    }
} 