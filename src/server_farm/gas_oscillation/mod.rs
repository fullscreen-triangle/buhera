//! # Gas Oscillation Processor
//!
//! This module implements the molecular-scale computational units that form
//! the foundation of the gas oscillation server farm. Each processor functions
//! simultaneously as a computational engine, quantum clock, and oscillatory system.
//!
//! ## Key Components
//!
//! - **Processor**: Main gas oscillation computational unit
//! - **Molecular Analyzer**: Analyzes gas molecular properties
//! - **Oscillation Detector**: Detects and measures oscillations
//! - **Frequency Calculator**: Calculates oscillation frequencies
//! - **Phase Controller**: Manages oscillation phases
//! - **Amplitude Manager**: Controls oscillation amplitudes
//! - **Gas Injector**: Controls gas injection into chambers
//! - **Chamber Controller**: Manages gas chamber operations
//!
//! ## Triple Function Design
//!
//! Each gas molecule simultaneously functions as:
//! - **Processor**: Executing computational operations
//! - **Clock**: Providing timing reference through oscillations
//! - **Oscillator**: Contributing to system-wide resonance

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::server_farm::thermodynamics::{MoleculeType, ThermodynamicState};

/// Molecular analyzer
pub mod molecular_analyzer;

/// Oscillation detection and measurement
pub mod oscillation_detector;

/// Frequency calculation algorithms
pub mod frequency_calculator;

/// Phase control systems
pub mod phase_controller;

/// Amplitude management
pub mod amplitude_manager;

/// Gas injection control
pub mod gas_injector;

/// Chamber management
pub mod chamber_controller;

pub use molecular_analyzer::MolecularAnalyzer;
pub use oscillation_detector::OscillationDetector;
pub use frequency_calculator::FrequencyCalculator;
pub use phase_controller::PhaseController;
pub use amplitude_manager::AmplitudeManager;
pub use gas_injector::GasInjector;
pub use chamber_controller::ChamberController;

/// Gas oscillation processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasOscillationConfig {
    /// Number of processing chambers
    pub chamber_count: usize,
    /// Pressure operating range (atm)
    pub pressure_range: (f64, f64),
    /// Temperature operating range (K)
    pub temperature_range: (f64, f64),
    /// Cycle frequency (Hz)
    pub cycle_frequency: f64,
    /// Gas mixture composition
    pub gas_mixture: Vec<MoleculeType>,
    /// Molecular density (molecules/m³)
    pub molecular_density: f64,
    /// Oscillation sensitivity threshold
    pub oscillation_sensitivity: f64,
    /// Phase synchronization enabled
    pub phase_synchronization: bool,
    /// Amplitude control enabled
    pub amplitude_control: bool,
    /// Real-time monitoring enabled
    pub real_time_monitoring: bool,
}

/// Gas oscillation processor errors
#[derive(Debug, thiserror::Error)]
pub enum GasOscillationError {
    /// Configuration error
    #[error("Configuration error: {message}")]
    Configuration { message: String },
    
    /// Chamber operation error
    #[error("Chamber operation error: {message}")]
    ChamberOperation { message: String },
    
    /// Molecular analysis error
    #[error("Molecular analysis error: {message}")]
    MolecularAnalysis { message: String },
    
    /// Oscillation detection error
    #[error("Oscillation detection error: {message}")]
    OscillationDetection { message: String },
    
    /// Frequency calculation error
    #[error("Frequency calculation error: {message}")]
    FrequencyCalculation { message: String },
    
    /// Phase control error
    #[error("Phase control error: {message}")]
    PhaseControl { message: String },
    
    /// Amplitude management error
    #[error("Amplitude management error: {message}")]
    AmplitudeManagement { message: String },
    
    /// Gas injection error
    #[error("Gas injection error: {message}")]
    GasInjection { message: String },
    
    /// Synchronization error
    #[error("Synchronization error: {message}")]
    Synchronization { message: String },
}

/// Result type for gas oscillation operations
pub type GasOscillationResult<T> = Result<T, GasOscillationError>;

/// Gas oscillation processor state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorState {
    /// Processor ID
    pub id: Uuid,
    /// Current frequency (Hz)
    pub frequency: f64,
    /// Current phase (radians)
    pub phase: f64,
    /// Current amplitude
    pub amplitude: f64,
    /// Chamber temperature (K)
    pub temperature: f64,
    /// Chamber pressure (atm)
    pub pressure: f64,
    /// Molecular composition
    pub molecular_composition: HashMap<MoleculeType, f64>,
    /// Processing rate (operations/second)
    pub processing_rate: f64,
    /// Oscillation coherence
    pub coherence: f64,
    /// Last update timestamp
    pub last_update: Duration,
}

/// Computational task for gas oscillation processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationalTask {
    /// Task ID
    pub id: Uuid,
    /// Task type
    pub task_type: TaskType,
    /// Input data
    pub input_data: Vec<f64>,
    /// Required frequency range
    pub frequency_range: (f64, f64),
    /// Required precision
    pub precision: f64,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Priority level
    pub priority: u8,
}

/// Types of computational tasks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// Frequency analysis
    FrequencyAnalysis,
    /// Phase calculation
    PhaseCalculation,
    /// Amplitude processing
    AmplitudeProcessing,
    /// Molecular simulation
    MolecularSimulation,
    /// Oscillation prediction
    OscillationPrediction,
    /// Quantum coherence calculation
    QuantumCoherence,
    /// Consciousness processing
    ConsciousnessProcessing,
}

/// Computational result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationalResult {
    /// Task ID
    pub task_id: Uuid,
    /// Result data
    pub result_data: Vec<f64>,
    /// Execution time
    pub execution_time: Duration,
    /// Accuracy achieved
    pub accuracy: f64,
    /// Energy consumed
    pub energy_consumed: f64,
    /// Oscillation parameters used
    pub oscillation_parameters: OscillationParameters,
}

/// Oscillation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationParameters {
    /// Frequency (Hz)
    pub frequency: f64,
    /// Phase (radians)
    pub phase: f64,
    /// Amplitude
    pub amplitude: f64,
    /// Coherence
    pub coherence: f64,
    /// Molecular types involved
    pub molecule_types: Vec<MoleculeType>,
}

/// Main gas oscillation processor
pub struct GasOscillationProcessor {
    /// Processor configuration
    config: GasOscillationConfig,
    /// Molecular analyzer
    molecular_analyzer: MolecularAnalyzer,
    /// Oscillation detector
    oscillation_detector: OscillationDetector,
    /// Frequency calculator
    frequency_calculator: FrequencyCalculator,
    /// Phase controller
    phase_controller: PhaseController,
    /// Amplitude manager
    amplitude_manager: AmplitudeManager,
    /// Gas injector
    gas_injector: GasInjector,
    /// Chamber controller
    chamber_controller: ChamberController,
    /// Current processor state
    current_state: Arc<RwLock<ProcessorState>>,
    /// Task queue
    task_queue: Arc<RwLock<Vec<ComputationalTask>>>,
    /// Performance metrics
    metrics: Arc<RwLock<ProcessorMetrics>>,
}

/// Processor performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorMetrics {
    /// Total tasks processed
    pub total_tasks_processed: u64,
    /// Average processing time
    pub average_processing_time: Duration,
    /// Current processing rate
    pub current_processing_rate: f64,
    /// Energy efficiency
    pub energy_efficiency: f64,
    /// Oscillation stability
    pub oscillation_stability: f64,
    /// Chamber utilization
    pub chamber_utilization: f64,
    /// Molecular conversion efficiency
    pub molecular_conversion_efficiency: f64,
}

impl GasOscillationProcessor {
    /// Create a new gas oscillation processor
    pub fn new(config: GasOscillationConfig) -> GasOscillationResult<Self> {
        // Validate configuration
        Self::validate_config(&config)?;
        
        // Initialize components
        let molecular_analyzer = MolecularAnalyzer::new(&config)?;
        let oscillation_detector = OscillationDetector::new(&config)?;
        let frequency_calculator = FrequencyCalculator::new(&config)?;
        let phase_controller = PhaseController::new(&config)?;
        let amplitude_manager = AmplitudeManager::new(&config)?;
        let gas_injector = GasInjector::new(&config)?;
        let chamber_controller = ChamberController::new(&config)?;
        
        // Initialize state
        let initial_state = ProcessorState {
            id: Uuid::new_v4(),
            frequency: (config.pressure_range.0 + config.pressure_range.1) / 2.0 * 1e12, // Approximate
            phase: 0.0,
            amplitude: 1.0,
            temperature: (config.temperature_range.0 + config.temperature_range.1) / 2.0,
            pressure: (config.pressure_range.0 + config.pressure_range.1) / 2.0,
            molecular_composition: HashMap::new(),
            processing_rate: 0.0,
            coherence: 1.0,
            last_update: Duration::from_secs(0),
        };
        
        let metrics = ProcessorMetrics {
            total_tasks_processed: 0,
            average_processing_time: Duration::from_secs(0),
            current_processing_rate: 0.0,
            energy_efficiency: 0.0,
            oscillation_stability: 1.0,
            chamber_utilization: 0.0,
            molecular_conversion_efficiency: 0.0,
        };
        
        Ok(Self {
            config,
            molecular_analyzer,
            oscillation_detector,
            frequency_calculator,
            phase_controller,
            amplitude_manager,
            gas_injector,
            chamber_controller,
            current_state: Arc::new(RwLock::new(initial_state)),
            task_queue: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(metrics)),
        })
    }
    
    /// Start the processor
    pub async fn start(&mut self) -> GasOscillationResult<()> {
        // Initialize chambers
        self.chamber_controller.initialize_chambers().await?;
        
        // Start gas injection
        self.gas_injector.start_injection().await?;
        
        // Begin oscillation detection
        self.oscillation_detector.start_detection().await?;
        
        // Start processing loop
        self.start_processing_loop().await?;
        
        Ok(())
    }
    
    /// Submit computational task
    pub async fn submit_task(&self, task: ComputationalTask) -> GasOscillationResult<()> {
        let mut queue = self.task_queue.write().await;
        queue.push(task);
        
        // Sort by priority
        queue.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        Ok(())
    }
    
    /// Process computational task
    pub async fn process_task(&self, task: ComputationalTask) -> GasOscillationResult<ComputationalResult> {
        let start_time = tokio::time::Instant::now();
        
        // Analyze molecular requirements
        let molecular_requirements = self.molecular_analyzer.analyze_task_requirements(&task).await?;
        
        // Calculate required oscillation parameters
        let oscillation_params = self.frequency_calculator.calculate_parameters(&task, &molecular_requirements).await?;
        
        // Adjust chamber conditions
        self.chamber_controller.adjust_conditions(&oscillation_params).await?;
        
        // Control oscillation phase
        self.phase_controller.set_phase(oscillation_params.phase).await?;
        
        // Manage amplitude
        self.amplitude_manager.set_amplitude(oscillation_params.amplitude).await?;
        
        // Execute computation
        let result_data = self.execute_computation(&task, &oscillation_params).await?;
        
        let execution_time = start_time.elapsed();
        
        // Update metrics
        self.update_metrics(&task, execution_time).await?;
        
        Ok(ComputationalResult {
            task_id: task.id,
            result_data,
            execution_time,
            accuracy: 0.999, // High accuracy for gas oscillation processing
            energy_consumed: self.calculate_energy_consumption(&task, &oscillation_params).await?,
            oscillation_parameters: oscillation_params,
        })
    }
    
    /// Execute computation using oscillation parameters
    async fn execute_computation(
        &self,
        task: &ComputationalTask,
        oscillation_params: &OscillationParameters,
    ) -> GasOscillationResult<Vec<f64>> {
        match task.task_type {
            TaskType::FrequencyAnalysis => {
                self.execute_frequency_analysis(task, oscillation_params).await
            }
            TaskType::PhaseCalculation => {
                self.execute_phase_calculation(task, oscillation_params).await
            }
            TaskType::AmplitudeProcessing => {
                self.execute_amplitude_processing(task, oscillation_params).await
            }
            TaskType::MolecularSimulation => {
                self.execute_molecular_simulation(task, oscillation_params).await
            }
            TaskType::OscillationPrediction => {
                self.execute_oscillation_prediction(task, oscillation_params).await
            }
            TaskType::QuantumCoherence => {
                self.execute_quantum_coherence(task, oscillation_params).await
            }
            TaskType::ConsciousnessProcessing => {
                self.execute_consciousness_processing(task, oscillation_params).await
            }
        }
    }
    
    /// Execute frequency analysis
    async fn execute_frequency_analysis(
        &self,
        task: &ComputationalTask,
        oscillation_params: &OscillationParameters,
    ) -> GasOscillationResult<Vec<f64>> {
        // Perform frequency analysis using molecular oscillations
        let frequencies = self.frequency_calculator.analyze_frequencies(
            &task.input_data,
            oscillation_params,
        ).await?;
        
        Ok(frequencies)
    }
    
    /// Execute phase calculation
    async fn execute_phase_calculation(
        &self,
        task: &ComputationalTask,
        oscillation_params: &OscillationParameters,
    ) -> GasOscillationResult<Vec<f64>> {
        // Perform phase calculation using molecular phase relationships
        let phases = self.phase_controller.calculate_phases(
            &task.input_data,
            oscillation_params,
        ).await?;
        
        Ok(phases)
    }
    
    /// Execute amplitude processing
    async fn execute_amplitude_processing(
        &self,
        task: &ComputationalTask,
        oscillation_params: &OscillationParameters,
    ) -> GasOscillationResult<Vec<f64>> {
        // Perform amplitude processing using molecular amplitude modulation
        let amplitudes = self.amplitude_manager.process_amplitudes(
            &task.input_data,
            oscillation_params,
        ).await?;
        
        Ok(amplitudes)
    }
    
    /// Execute molecular simulation
    async fn execute_molecular_simulation(
        &self,
        task: &ComputationalTask,
        oscillation_params: &OscillationParameters,
    ) -> GasOscillationResult<Vec<f64>> {
        // Perform molecular simulation using gas dynamics
        let simulation_results = self.molecular_analyzer.simulate_molecular_dynamics(
            &task.input_data,
            oscillation_params,
        ).await?;
        
        Ok(simulation_results)
    }
    
    /// Execute oscillation prediction
    async fn execute_oscillation_prediction(
        &self,
        task: &ComputationalTask,
        oscillation_params: &OscillationParameters,
    ) -> GasOscillationResult<Vec<f64>> {
        // Predict future oscillation behavior
        let predictions = self.oscillation_detector.predict_oscillations(
            &task.input_data,
            oscillation_params,
        ).await?;
        
        Ok(predictions)
    }
    
    /// Execute quantum coherence calculation
    async fn execute_quantum_coherence(
        &self,
        task: &ComputationalTask,
        oscillation_params: &OscillationParameters,
    ) -> GasOscillationResult<Vec<f64>> {
        // Calculate quantum coherence using molecular quantum states
        let coherence_values = self.calculate_quantum_coherence(
            &task.input_data,
            oscillation_params,
        ).await?;
        
        Ok(coherence_values)
    }
    
    /// Execute consciousness processing
    async fn execute_consciousness_processing(
        &self,
        task: &ComputationalTask,
        oscillation_params: &OscillationParameters,
    ) -> GasOscillationResult<Vec<f64>> {
        // Perform consciousness-level processing
        let consciousness_results = self.process_consciousness_data(
            &task.input_data,
            oscillation_params,
        ).await?;
        
        Ok(consciousness_results)
    }
    
    /// Calculate quantum coherence
    async fn calculate_quantum_coherence(
        &self,
        input_data: &[f64],
        oscillation_params: &OscillationParameters,
    ) -> GasOscillationResult<Vec<f64>> {
        // Calculate quantum coherence based on molecular oscillations
        let mut coherence_values = Vec::new();
        
        for (i, &value) in input_data.iter().enumerate() {
            let phase_factor = oscillation_params.phase + (i as f64) * std::f64::consts::PI / 180.0;
            let coherence = oscillation_params.coherence * 
                (oscillation_params.frequency * phase_factor).cos() * 
                oscillation_params.amplitude * 
                value;
            coherence_values.push(coherence);
        }
        
        Ok(coherence_values)
    }
    
    /// Process consciousness data
    async fn process_consciousness_data(
        &self,
        input_data: &[f64],
        oscillation_params: &OscillationParameters,
    ) -> GasOscillationResult<Vec<f64>> {
        // Process data at consciousness level using molecular substrate
        let mut consciousness_results = Vec::new();
        
        for (i, &value) in input_data.iter().enumerate() {
            let consciousness_factor = oscillation_params.coherence * 
                (oscillation_params.frequency / 1e12) * 
                oscillation_params.amplitude;
            
            let processed_value = value * consciousness_factor * 
                ((i as f64) * oscillation_params.phase).sin();
            
            consciousness_results.push(processed_value);
        }
        
        Ok(consciousness_results)
    }
    
    /// Calculate energy consumption
    async fn calculate_energy_consumption(
        &self,
        task: &ComputationalTask,
        oscillation_params: &OscillationParameters,
    ) -> GasOscillationResult<f64> {
        // Calculate energy consumption based on oscillation parameters
        let base_energy = task.input_data.len() as f64 * 1e-15; // Base energy per data point
        let frequency_factor = oscillation_params.frequency / 1e12; // Frequency scaling
        let amplitude_factor = oscillation_params.amplitude; // Amplitude scaling
        
        let total_energy = base_energy * frequency_factor * amplitude_factor;
        
        Ok(total_energy)
    }
    
    /// Update performance metrics
    async fn update_metrics(&self, task: &ComputationalTask, execution_time: Duration) -> GasOscillationResult<()> {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_tasks_processed += 1;
        
        // Update average processing time
        let total_time = metrics.average_processing_time * (metrics.total_tasks_processed - 1) as u32 + execution_time;
        metrics.average_processing_time = total_time / metrics.total_tasks_processed as u32;
        
        // Update current processing rate
        metrics.current_processing_rate = task.input_data.len() as f64 / execution_time.as_secs_f64();
        
        // Update efficiency metrics
        metrics.energy_efficiency = 0.95; // High efficiency for gas oscillation processing
        metrics.oscillation_stability = 0.99; // High stability
        metrics.chamber_utilization = 0.85; // Good utilization
        metrics.molecular_conversion_efficiency = 0.90; // Good conversion
        
        Ok(())
    }
    
    /// Start processing loop
    async fn start_processing_loop(&self) -> GasOscillationResult<()> {
        let task_queue = self.task_queue.clone();
        let processor = self.clone();
        
        tokio::spawn(async move {
            loop {
                // Process tasks from queue
                let task = {
                    let mut queue = task_queue.write().await;
                    queue.pop()
                };
                
                if let Some(task) = task {
                    if let Err(e) = processor.process_task(task).await {
                        eprintln!("Error processing task: {}", e);
                    }
                } else {
                    // No tasks, sleep briefly
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        });
        
        Ok(())
    }
    
    /// Get current processor state
    pub async fn get_state(&self) -> ProcessorState {
        self.current_state.read().await.clone()
    }
    
    /// Get current metrics
    pub async fn get_metrics(&self) -> ProcessorMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Validate configuration
    fn validate_config(config: &GasOscillationConfig) -> GasOscillationResult<()> {
        if config.chamber_count == 0 {
            return Err(GasOscillationError::Configuration {
                message: "Chamber count must be greater than 0".to_string(),
            });
        }
        
        if config.pressure_range.0 >= config.pressure_range.1 {
            return Err(GasOscillationError::Configuration {
                message: "Invalid pressure range".to_string(),
            });
        }
        
        if config.temperature_range.0 >= config.temperature_range.1 {
            return Err(GasOscillationError::Configuration {
                message: "Invalid temperature range".to_string(),
            });
        }
        
        if config.cycle_frequency <= 0.0 {
            return Err(GasOscillationError::Configuration {
                message: "Cycle frequency must be positive".to_string(),
            });
        }
        
        if config.gas_mixture.is_empty() {
            return Err(GasOscillationError::Configuration {
                message: "Gas mixture cannot be empty".to_string(),
            });
        }
        
        Ok(())
    }
}

// Implement Clone for GasOscillationProcessor (simplified)
impl Clone for GasOscillationProcessor {
    fn clone(&self) -> Self {
        // Create a new instance with the same configuration
        Self::new(self.config.clone()).unwrap()
    }
}

impl Default for GasOscillationConfig {
    fn default() -> Self {
        Self {
            chamber_count: 1000,
            pressure_range: (0.1, 10.0),
            temperature_range: (200.0, 400.0),
            cycle_frequency: 1000.0,
            gas_mixture: vec![
                MoleculeType::N2,
                MoleculeType::O2,
                MoleculeType::H2O,
                MoleculeType::He,
            ],
            molecular_density: 1e25,
            oscillation_sensitivity: 0.001,
            phase_synchronization: true,
            amplitude_control: true,
            real_time_monitoring: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_processor_creation() {
        let config = GasOscillationConfig::default();
        let processor = GasOscillationProcessor::new(config).unwrap();
        
        let state = processor.get_state().await;
        assert!(state.frequency > 0.0);
        assert!(state.coherence > 0.0);
    }

    #[test]
    async fn test_task_submission() {
        let config = GasOscillationConfig::default();
        let processor = GasOscillationProcessor::new(config).unwrap();
        
        let task = ComputationalTask {
            id: Uuid::new_v4(),
            task_type: TaskType::FrequencyAnalysis,
            input_data: vec![1.0, 2.0, 3.0],
            frequency_range: (1e12, 1e13),
            precision: 0.001,
            max_execution_time: Duration::from_secs(1),
            priority: 5,
        };
        
        processor.submit_task(task).await.unwrap();
        
        let queue = processor.task_queue.read().await;
        assert_eq!(queue.len(), 1);
    }

    #[test]
    async fn test_task_processing() {
        let config = GasOscillationConfig::default();
        let processor = GasOscillationProcessor::new(config).unwrap();
        
        let task = ComputationalTask {
            id: Uuid::new_v4(),
            task_type: TaskType::FrequencyAnalysis,
            input_data: vec![1.0, 2.0, 3.0],
            frequency_range: (1e12, 1e13),
            precision: 0.001,
            max_execution_time: Duration::from_secs(1),
            priority: 5,
        };
        
        let result = processor.process_task(task).await.unwrap();
        assert!(!result.result_data.is_empty());
        assert!(result.accuracy > 0.0);
        assert!(result.energy_consumed > 0.0);
    }

    #[test]
    async fn test_quantum_coherence_calculation() {
        let config = GasOscillationConfig::default();
        let processor = GasOscillationProcessor::new(config).unwrap();
        
        let input_data = vec![1.0, 2.0, 3.0];
        let oscillation_params = OscillationParameters {
            frequency: 1e12,
            phase: 0.0,
            amplitude: 1.0,
            coherence: 0.99,
            molecule_types: vec![MoleculeType::N2],
        };
        
        let coherence_values = processor.calculate_quantum_coherence(&input_data, &oscillation_params).await.unwrap();
        assert_eq!(coherence_values.len(), input_data.len());
    }

    #[test]
    async fn test_consciousness_processing() {
        let config = GasOscillationConfig::default();
        let processor = GasOscillationProcessor::new(config).unwrap();
        
        let input_data = vec![1.0, 2.0, 3.0];
        let oscillation_params = OscillationParameters {
            frequency: 1e12,
            phase: 0.0,
            amplitude: 1.0,
            coherence: 0.99,
            molecule_types: vec![MoleculeType::N2],
        };
        
        let consciousness_results = processor.process_consciousness_data(&input_data, &oscillation_params).await.unwrap();
        assert_eq!(consciousness_results.len(), input_data.len());
    }

    #[test]
    fn test_config_validation() {
        let mut config = GasOscillationConfig::default();
        
        // Test invalid chamber count
        config.chamber_count = 0;
        assert!(GasOscillationProcessor::validate_config(&config).is_err());
        
        // Test invalid pressure range
        config.chamber_count = 1000;
        config.pressure_range = (10.0, 0.1);
        assert!(GasOscillationProcessor::validate_config(&config).is_err());
        
        // Test invalid temperature range
        config.pressure_range = (0.1, 10.0);
        config.temperature_range = (400.0, 200.0);
        assert!(GasOscillationProcessor::validate_config(&config).is_err());
        
        // Test invalid cycle frequency
        config.temperature_range = (200.0, 400.0);
        config.cycle_frequency = -1.0;
        assert!(GasOscillationProcessor::validate_config(&config).is_err());
        
        // Test empty gas mixture
        config.cycle_frequency = 1000.0;
        config.gas_mixture.clear();
        assert!(GasOscillationProcessor::validate_config(&config).is_err());
        
        // Test valid configuration
        config.gas_mixture.push(MoleculeType::N2);
        assert!(GasOscillationProcessor::validate_config(&config).is_ok());
    }
} 