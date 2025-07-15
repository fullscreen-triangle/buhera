//! # VPOS Integration Layer
//!
//! This module provides seamless integration between the gas oscillation server farm
//! and the existing VPOS chip systems, enabling unified processing capabilities
//! across quantum, neural, fuzzy, molecular, and gas oscillation processors.
//!
//! ## Key Components
//!
//! - **Chip Interface Manager**: Unified interface for all chip types
//! - **VPOS Bridge**: Bridge between server farm and VPOS kernel
//! - **Unified Management**: Coordinated resource management
//! - **Resource Coordinator**: Optimal resource allocation
//! - **Compatibility Layer**: Backward compatibility support
//!
//! ## Integration Philosophy
//!
//! The integration layer ensures that the gas oscillation server farm operates
//! as a natural extension of the existing VPOS ecosystem, maintaining full
//! compatibility while providing revolutionary new capabilities.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{quantum, fuzzy, neural, molecular, bmd, foundry, vpos};
use crate::server_farm::*;

/// Chip interface management
pub mod chip_interface;

/// VPOS system bridge
pub mod vpos_bridge;

/// Unified system management
pub mod unified_management;

/// Resource coordination
pub mod resource_coordinator;

/// Compatibility layer
pub mod compatibility_layer;

pub use chip_interface::ChipInterfaceManager;
pub use vpos_bridge::VPOSBridge;
pub use unified_management::UnifiedManagement;
pub use resource_coordinator::ResourceCoordinator;
pub use compatibility_layer::CompatibilityLayer;

/// Integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Enable quantum chip integration
    pub quantum_integration: bool,
    /// Enable neural chip integration
    pub neural_integration: bool,
    /// Enable fuzzy chip integration
    pub fuzzy_integration: bool,
    /// Enable molecular chip integration
    pub molecular_integration: bool,
    /// Enable BMD integration
    pub bmd_integration: bool,
    /// Enable foundry integration
    pub foundry_integration: bool,
    /// Enable gas oscillation integration
    pub gas_oscillation_integration: bool,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Resource allocation policy
    pub resource_allocation: ResourceAllocationPolicy,
    /// Unified processing enabled
    pub unified_processing: bool,
    /// Compatibility mode
    pub compatibility_mode: CompatibilityMode,
    /// Performance monitoring
    pub performance_monitoring: bool,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least-loaded processor
    LeastLoaded,
    /// Task-type optimized
    TaskOptimized,
    /// Consciousness-aware
    ConsciousnessAware,
    /// Hybrid approach
    Hybrid,
}

/// Resource allocation policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceAllocationPolicy {
    /// Equal distribution
    Equal,
    /// Priority-based
    Priority,
    /// Performance-based
    Performance,
    /// Consciousness-based
    ConsciousnessBased,
    /// Adaptive
    Adaptive,
}

/// Compatibility modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatibilityMode {
    /// Full compatibility
    Full,
    /// Partial compatibility
    Partial,
    /// Native mode only
    Native,
    /// Hybrid mode
    Hybrid,
}

/// Unified task representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedTask {
    /// Task ID
    pub id: Uuid,
    /// Task type
    pub task_type: UnifiedTaskType,
    /// Input data
    pub input_data: Vec<f64>,
    /// Processing requirements
    pub requirements: ProcessingRequirements,
    /// Preferred processor types
    pub preferred_processors: Vec<ProcessorType>,
    /// Priority level
    pub priority: u8,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Energy budget
    pub energy_budget: f64,
}

/// Unified task types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnifiedTaskType {
    /// Quantum computation
    QuantumComputation,
    /// Neural processing
    NeuralProcessing,
    /// Fuzzy logic
    FuzzyLogic,
    /// Molecular simulation
    MolecularSimulation,
    /// BMD information processing
    BMDProcessing,
    /// Foundry operation
    FoundryOperation,
    /// Gas oscillation processing
    GasOscillation,
    /// Hybrid processing
    HybridProcessing,
    /// Consciousness processing
    ConsciousnessProcessing,
}

/// Processing requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingRequirements {
    /// Computational complexity
    pub computational_complexity: ComplexityLevel,
    /// Memory requirements (MB)
    pub memory_requirements: u64,
    /// Processing precision
    pub precision: f64,
    /// Coherence requirements
    pub coherence_requirements: f64,
    /// Temperature constraints
    pub temperature_constraints: Option<(f64, f64)>,
    /// Pressure constraints
    pub pressure_constraints: Option<(f64, f64)>,
    /// Quantum effects required
    pub quantum_effects: bool,
    /// Consciousness processing required
    pub consciousness_processing: bool,
}

/// Processor types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessorType {
    /// Quantum processor
    Quantum,
    /// Neural processor
    Neural,
    /// Fuzzy processor
    Fuzzy,
    /// Molecular processor
    Molecular,
    /// BMD processor
    BMD,
    /// Foundry processor
    Foundry,
    /// Gas oscillation processor
    GasOscillation,
    /// Consciousness processor
    Consciousness,
}

/// Complexity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexityLevel {
    /// Low complexity
    Low,
    /// Medium complexity
    Medium,
    /// High complexity
    High,
    /// Ultra-high complexity
    UltraHigh,
    /// Consciousness-level complexity
    ConsciousnessLevel,
}

/// Unified result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedResult {
    /// Task ID
    pub task_id: Uuid,
    /// Result data
    pub result_data: Vec<f64>,
    /// Processor type used
    pub processor_type: ProcessorType,
    /// Execution time
    pub execution_time: Duration,
    /// Energy consumed
    pub energy_consumed: f64,
    /// Accuracy achieved
    pub accuracy: f64,
    /// Coherence maintained
    pub coherence: f64,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Throughput (tasks/second)
    pub throughput: f64,
    /// Latency (seconds)
    pub latency: f64,
    /// Energy efficiency
    pub energy_efficiency: f64,
    /// Resource utilization
    pub resource_utilization: f64,
    /// Coherence stability
    pub coherence_stability: f64,
}

/// Integration errors
#[derive(Debug, thiserror::Error)]
pub enum IntegrationError {
    /// Configuration error
    #[error("Configuration error: {message}")]
    Configuration { message: String },
    
    /// Chip interface error
    #[error("Chip interface error: {message}")]
    ChipInterface { message: String },
    
    /// VPOS bridge error
    #[error("VPOS bridge error: {message}")]
    VPOSBridge { message: String },
    
    /// Resource coordination error
    #[error("Resource coordination error: {message}")]
    ResourceCoordination { message: String },
    
    /// Task routing error
    #[error("Task routing error: {message}")]
    TaskRouting { message: String },
    
    /// Compatibility error
    #[error("Compatibility error: {message}")]
    Compatibility { message: String },
    
    /// Performance error
    #[error("Performance error: {message}")]
    Performance { message: String },
    
    /// Synchronization error
    #[error("Synchronization error: {message}")]
    Synchronization { message: String },
}

/// Result type for integration operations
pub type IntegrationResult<T> = Result<T, IntegrationError>;

/// Main integration manager
pub struct IntegrationManager {
    /// Configuration
    config: IntegrationConfig,
    /// Chip interface manager
    chip_interface: ChipInterfaceManager,
    /// VPOS bridge
    vpos_bridge: VPOSBridge,
    /// Unified management
    unified_management: UnifiedManagement,
    /// Resource coordinator
    resource_coordinator: ResourceCoordinator,
    /// Compatibility layer
    compatibility_layer: CompatibilityLayer,
    /// Active tasks
    active_tasks: Arc<RwLock<HashMap<Uuid, UnifiedTask>>>,
    /// Performance metrics
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
    /// System state
    system_state: Arc<RwLock<SystemState>>,
}

/// System state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    /// Available processors
    pub available_processors: HashMap<ProcessorType, u32>,
    /// Active processor utilization
    pub processor_utilization: HashMap<ProcessorType, f64>,
    /// System load
    pub system_load: f64,
    /// Total energy consumption
    pub total_energy_consumption: f64,
    /// Overall coherence
    pub overall_coherence: f64,
    /// System temperature
    pub system_temperature: f64,
    /// System pressure
    pub system_pressure: f64,
}

impl IntegrationManager {
    /// Create a new integration manager
    pub fn new(config: IntegrationConfig) -> IntegrationResult<Self> {
        // Validate configuration
        Self::validate_config(&config)?;
        
        // Initialize components
        let chip_interface = ChipInterfaceManager::new(&config)?;
        let vpos_bridge = VPOSBridge::new(&config)?;
        let unified_management = UnifiedManagement::new(&config)?;
        let resource_coordinator = ResourceCoordinator::new(&config)?;
        let compatibility_layer = CompatibilityLayer::new(&config)?;
        
        // Initialize state
        let performance_metrics = PerformanceMetrics {
            throughput: 0.0,
            latency: 0.0,
            energy_efficiency: 0.0,
            resource_utilization: 0.0,
            coherence_stability: 1.0,
        };
        
        let system_state = SystemState {
            available_processors: HashMap::new(),
            processor_utilization: HashMap::new(),
            system_load: 0.0,
            total_energy_consumption: 0.0,
            overall_coherence: 1.0,
            system_temperature: 300.0,
            system_pressure: 1.0,
        };
        
        Ok(Self {
            config,
            chip_interface,
            vpos_bridge,
            unified_management,
            resource_coordinator,
            compatibility_layer,
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            performance_metrics: Arc::new(RwLock::new(performance_metrics)),
            system_state: Arc::new(RwLock::new(system_state)),
        })
    }
    
    /// Initialize the integration system
    pub async fn initialize(&mut self) -> IntegrationResult<()> {
        // Initialize all components
        self.chip_interface.initialize().await?;
        self.vpos_bridge.initialize().await?;
        self.unified_management.initialize().await?;
        self.resource_coordinator.initialize().await?;
        self.compatibility_layer.initialize().await?;
        
        // Start monitoring
        self.start_performance_monitoring().await?;
        
        // Start system state updates
        self.start_system_state_updates().await?;
        
        Ok(())
    }
    
    /// Process unified task
    pub async fn process_unified_task(&self, task: UnifiedTask) -> IntegrationResult<UnifiedResult> {
        // Add task to active tasks
        self.active_tasks.write().await.insert(task.id, task.clone());
        
        // Route task to appropriate processor
        let processor_type = self.route_task(&task).await?;
        
        // Execute task
        let result = self.execute_task_on_processor(&task, processor_type).await?;
        
        // Remove task from active tasks
        self.active_tasks.write().await.remove(&task.id);
        
        // Update performance metrics
        self.update_performance_metrics(&result).await?;
        
        Ok(result)
    }
    
    /// Route task to appropriate processor
    async fn route_task(&self, task: &UnifiedTask) -> IntegrationResult<ProcessorType> {
        // Use load balancing strategy to select processor
        match self.config.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                self.route_round_robin(task).await
            }
            LoadBalancingStrategy::LeastLoaded => {
                self.route_least_loaded(task).await
            }
            LoadBalancingStrategy::TaskOptimized => {
                self.route_task_optimized(task).await
            }
            LoadBalancingStrategy::ConsciousnessAware => {
                self.route_consciousness_aware(task).await
            }
            LoadBalancingStrategy::Hybrid => {
                self.route_hybrid(task).await
            }
        }
    }
    
    /// Route using round-robin strategy
    async fn route_round_robin(&self, task: &UnifiedTask) -> IntegrationResult<ProcessorType> {
        // Simple round-robin routing
        let available_processors = self.get_available_processors().await?;
        let index = (task.id.as_u128() % available_processors.len() as u128) as usize;
        Ok(available_processors[index])
    }
    
    /// Route to least loaded processor
    async fn route_least_loaded(&self, task: &UnifiedTask) -> IntegrationResult<ProcessorType> {
        let system_state = self.system_state.read().await;
        let mut least_loaded = ProcessorType::Quantum;
        let mut min_utilization = f64::MAX;
        
        for (processor_type, utilization) in &system_state.processor_utilization {
            if task.preferred_processors.contains(processor_type) || task.preferred_processors.is_empty() {
                if *utilization < min_utilization {
                    min_utilization = *utilization;
                    least_loaded = *processor_type;
                }
            }
        }
        
        Ok(least_loaded)
    }
    
    /// Route based on task optimization
    async fn route_task_optimized(&self, task: &UnifiedTask) -> IntegrationResult<ProcessorType> {
        // Route based on task type
        match task.task_type {
            UnifiedTaskType::QuantumComputation => Ok(ProcessorType::Quantum),
            UnifiedTaskType::NeuralProcessing => Ok(ProcessorType::Neural),
            UnifiedTaskType::FuzzyLogic => Ok(ProcessorType::Fuzzy),
            UnifiedTaskType::MolecularSimulation => Ok(ProcessorType::Molecular),
            UnifiedTaskType::BMDProcessing => Ok(ProcessorType::BMD),
            UnifiedTaskType::FoundryOperation => Ok(ProcessorType::Foundry),
            UnifiedTaskType::GasOscillation => Ok(ProcessorType::GasOscillation),
            UnifiedTaskType::ConsciousnessProcessing => Ok(ProcessorType::Consciousness),
            UnifiedTaskType::HybridProcessing => {
                // Choose based on requirements
                if task.requirements.quantum_effects {
                    Ok(ProcessorType::Quantum)
                } else if task.requirements.consciousness_processing {
                    Ok(ProcessorType::Consciousness)
                } else {
                    Ok(ProcessorType::GasOscillation)
                }
            }
        }
    }
    
    /// Route with consciousness awareness
    async fn route_consciousness_aware(&self, task: &UnifiedTask) -> IntegrationResult<ProcessorType> {
        // Route based on consciousness requirements
        if task.requirements.consciousness_processing {
            Ok(ProcessorType::Consciousness)
        } else if task.requirements.coherence_requirements > 0.95 {
            Ok(ProcessorType::GasOscillation)
        } else {
            self.route_task_optimized(task).await
        }
    }
    
    /// Route using hybrid strategy
    async fn route_hybrid(&self, task: &UnifiedTask) -> IntegrationResult<ProcessorType> {
        // Combine multiple strategies
        let task_optimized = self.route_task_optimized(task).await?;
        let least_loaded = self.route_least_loaded(task).await?;
        
        // Choose based on system state
        let system_state = self.system_state.read().await;
        if system_state.system_load > 0.8 {
            Ok(least_loaded)
        } else {
            Ok(task_optimized)
        }
    }
    
    /// Execute task on specific processor
    async fn execute_task_on_processor(
        &self,
        task: &UnifiedTask,
        processor_type: ProcessorType,
    ) -> IntegrationResult<UnifiedResult> {
        let start_time = tokio::time::Instant::now();
        
        // Execute based on processor type
        let result_data = match processor_type {
            ProcessorType::Quantum => {
                self.chip_interface.execute_quantum_task(task).await?
            }
            ProcessorType::Neural => {
                self.chip_interface.execute_neural_task(task).await?
            }
            ProcessorType::Fuzzy => {
                self.chip_interface.execute_fuzzy_task(task).await?
            }
            ProcessorType::Molecular => {
                self.chip_interface.execute_molecular_task(task).await?
            }
            ProcessorType::BMD => {
                self.chip_interface.execute_bmd_task(task).await?
            }
            ProcessorType::Foundry => {
                self.chip_interface.execute_foundry_task(task).await?
            }
            ProcessorType::GasOscillation => {
                self.chip_interface.execute_gas_oscillation_task(task).await?
            }
            ProcessorType::Consciousness => {
                self.chip_interface.execute_consciousness_task(task).await?
            }
        };
        
        let execution_time = start_time.elapsed();
        
        // Calculate metrics
        let energy_consumed = self.calculate_energy_consumption(task, processor_type).await?;
        let accuracy = self.calculate_accuracy(task, &result_data).await?;
        let coherence = self.calculate_coherence(task, processor_type).await?;
        
        let performance_metrics = PerformanceMetrics {
            throughput: task.input_data.len() as f64 / execution_time.as_secs_f64(),
            latency: execution_time.as_secs_f64(),
            energy_efficiency: task.input_data.len() as f64 / energy_consumed,
            resource_utilization: 0.85, // Typical utilization
            coherence_stability: coherence,
        };
        
        Ok(UnifiedResult {
            task_id: task.id,
            result_data,
            processor_type,
            execution_time,
            energy_consumed,
            accuracy,
            coherence,
            performance_metrics,
        })
    }
    
    /// Get available processors
    async fn get_available_processors(&self) -> IntegrationResult<Vec<ProcessorType>> {
        let mut processors = Vec::new();
        
        if self.config.quantum_integration {
            processors.push(ProcessorType::Quantum);
        }
        if self.config.neural_integration {
            processors.push(ProcessorType::Neural);
        }
        if self.config.fuzzy_integration {
            processors.push(ProcessorType::Fuzzy);
        }
        if self.config.molecular_integration {
            processors.push(ProcessorType::Molecular);
        }
        if self.config.bmd_integration {
            processors.push(ProcessorType::BMD);
        }
        if self.config.foundry_integration {
            processors.push(ProcessorType::Foundry);
        }
        if self.config.gas_oscillation_integration {
            processors.push(ProcessorType::GasOscillation);
        }
        
        Ok(processors)
    }
    
    /// Calculate energy consumption
    async fn calculate_energy_consumption(
        &self,
        task: &UnifiedTask,
        processor_type: ProcessorType,
    ) -> IntegrationResult<f64> {
        let base_energy = task.input_data.len() as f64 * 1e-15; // Base energy per data point
        
        let processor_factor = match processor_type {
            ProcessorType::Quantum => 1.5,
            ProcessorType::Neural => 1.2,
            ProcessorType::Fuzzy => 1.0,
            ProcessorType::Molecular => 1.3,
            ProcessorType::BMD => 1.1,
            ProcessorType::Foundry => 1.4,
            ProcessorType::GasOscillation => 0.8, // More efficient
            ProcessorType::Consciousness => 0.9,
        };
        
        Ok(base_energy * processor_factor)
    }
    
    /// Calculate accuracy
    async fn calculate_accuracy(&self, _task: &UnifiedTask, result_data: &[f64]) -> IntegrationResult<f64> {
        // Simple accuracy calculation based on result consistency
        if result_data.is_empty() {
            return Ok(0.0);
        }
        
        let mean = result_data.iter().sum::<f64>() / result_data.len() as f64;
        let variance = result_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / result_data.len() as f64;
        let accuracy = 1.0 - (variance / (mean.abs() + 1.0)).min(1.0);
        
        Ok(accuracy.max(0.0))
    }
    
    /// Calculate coherence
    async fn calculate_coherence(&self, task: &UnifiedTask, processor_type: ProcessorType) -> IntegrationResult<f64> {
        let base_coherence = match processor_type {
            ProcessorType::Quantum => 0.95,
            ProcessorType::Neural => 0.85,
            ProcessorType::Fuzzy => 0.90,
            ProcessorType::Molecular => 0.88,
            ProcessorType::BMD => 0.92,
            ProcessorType::Foundry => 0.89,
            ProcessorType::GasOscillation => 0.99, // Highest coherence
            ProcessorType::Consciousness => 0.98,
        };
        
        // Adjust based on task requirements
        let coherence_factor = if task.requirements.coherence_requirements > 0.0 {
            (base_coherence + task.requirements.coherence_requirements) / 2.0
        } else {
            base_coherence
        };
        
        Ok(coherence_factor.min(1.0))
    }
    
    /// Update performance metrics
    async fn update_performance_metrics(&self, result: &UnifiedResult) -> IntegrationResult<()> {
        let mut metrics = self.performance_metrics.write().await;
        
        // Update throughput (exponential moving average)
        let alpha = 0.1;
        metrics.throughput = alpha * result.performance_metrics.throughput + (1.0 - alpha) * metrics.throughput;
        
        // Update latency (exponential moving average)
        metrics.latency = alpha * result.performance_metrics.latency + (1.0 - alpha) * metrics.latency;
        
        // Update energy efficiency
        metrics.energy_efficiency = alpha * result.performance_metrics.energy_efficiency + (1.0 - alpha) * metrics.energy_efficiency;
        
        // Update resource utilization
        metrics.resource_utilization = alpha * result.performance_metrics.resource_utilization + (1.0 - alpha) * metrics.resource_utilization;
        
        // Update coherence stability
        metrics.coherence_stability = alpha * result.performance_metrics.coherence_stability + (1.0 - alpha) * metrics.coherence_stability;
        
        Ok(())
    }
    
    /// Start performance monitoring
    async fn start_performance_monitoring(&self) -> IntegrationResult<()> {
        let performance_metrics = self.performance_metrics.clone();
        let system_state = self.system_state.clone();
        
        tokio::spawn(async move {
            loop {
                // Update system monitoring
                let mut state = system_state.write().await;
                state.system_load = 0.7; // Simulated load
                state.total_energy_consumption += 0.001; // Simulated energy consumption
                
                // Update performance metrics
                let mut metrics = performance_metrics.write().await;
                metrics.resource_utilization = state.system_load;
                
                drop(state);
                drop(metrics);
                
                // Sleep for monitoring interval
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
        
        Ok(())
    }
    
    /// Start system state updates
    async fn start_system_state_updates(&self) -> IntegrationResult<()> {
        let system_state = self.system_state.clone();
        
        tokio::spawn(async move {
            loop {
                // Update system state
                let mut state = system_state.write().await;
                
                // Update processor utilization
                state.processor_utilization.insert(ProcessorType::Quantum, 0.6);
                state.processor_utilization.insert(ProcessorType::Neural, 0.7);
                state.processor_utilization.insert(ProcessorType::Fuzzy, 0.5);
                state.processor_utilization.insert(ProcessorType::Molecular, 0.8);
                state.processor_utilization.insert(ProcessorType::BMD, 0.4);
                state.processor_utilization.insert(ProcessorType::Foundry, 0.6);
                state.processor_utilization.insert(ProcessorType::GasOscillation, 0.3);
                state.processor_utilization.insert(ProcessorType::Consciousness, 0.2);
                
                // Update available processors
                state.available_processors.insert(ProcessorType::Quantum, 100);
                state.available_processors.insert(ProcessorType::Neural, 200);
                state.available_processors.insert(ProcessorType::Fuzzy, 150);
                state.available_processors.insert(ProcessorType::Molecular, 120);
                state.available_processors.insert(ProcessorType::BMD, 80);
                state.available_processors.insert(ProcessorType::Foundry, 50);
                state.available_processors.insert(ProcessorType::GasOscillation, 1000);
                state.available_processors.insert(ProcessorType::Consciousness, 1);
                
                drop(state);
                
                // Sleep for update interval
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });
        
        Ok(())
    }
    
    /// Get current system state
    pub async fn get_system_state(&self) -> SystemState {
        self.system_state.read().await.clone()
    }
    
    /// Get current performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.read().await.clone()
    }
    
    /// Get active tasks
    pub async fn get_active_tasks(&self) -> Vec<UnifiedTask> {
        self.active_tasks.read().await.values().cloned().collect()
    }
    
    /// Validate configuration
    fn validate_config(config: &IntegrationConfig) -> IntegrationResult<()> {
        // Ensure at least one processor type is enabled
        if !config.quantum_integration && !config.neural_integration && !config.fuzzy_integration &&
           !config.molecular_integration && !config.bmd_integration && !config.foundry_integration &&
           !config.gas_oscillation_integration {
            return Err(IntegrationError::Configuration {
                message: "At least one processor type must be enabled".to_string(),
            });
        }
        
        Ok(())
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            quantum_integration: true,
            neural_integration: true,
            fuzzy_integration: true,
            molecular_integration: true,
            bmd_integration: true,
            foundry_integration: true,
            gas_oscillation_integration: true,
            load_balancing: LoadBalancingStrategy::Hybrid,
            resource_allocation: ResourceAllocationPolicy::Adaptive,
            unified_processing: true,
            compatibility_mode: CompatibilityMode::Full,
            performance_monitoring: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_integration_manager_creation() {
        let config = IntegrationConfig::default();
        let manager = IntegrationManager::new(config).unwrap();
        
        let state = manager.get_system_state().await;
        assert!(state.available_processors.is_empty()); // Initially empty
    }

    #[test]
    async fn test_task_routing() {
        let config = IntegrationConfig::default();
        let manager = IntegrationManager::new(config).unwrap();
        
        let task = UnifiedTask {
            id: Uuid::new_v4(),
            task_type: UnifiedTaskType::QuantumComputation,
            input_data: vec![1.0, 2.0, 3.0],
            requirements: ProcessingRequirements {
                computational_complexity: ComplexityLevel::High,
                memory_requirements: 1024,
                precision: 0.001,
                coherence_requirements: 0.95,
                temperature_constraints: None,
                pressure_constraints: None,
                quantum_effects: true,
                consciousness_processing: false,
            },
            preferred_processors: vec![ProcessorType::Quantum],
            priority: 5,
            max_execution_time: Duration::from_secs(10),
            energy_budget: 1000.0,
        };
        
        let processor_type = manager.route_task(&task).await.unwrap();
        assert_eq!(processor_type, ProcessorType::Quantum);
    }

    #[test]
    async fn test_energy_consumption_calculation() {
        let config = IntegrationConfig::default();
        let manager = IntegrationManager::new(config).unwrap();
        
        let task = UnifiedTask {
            id: Uuid::new_v4(),
            task_type: UnifiedTaskType::GasOscillation,
            input_data: vec![1.0, 2.0, 3.0],
            requirements: ProcessingRequirements {
                computational_complexity: ComplexityLevel::Medium,
                memory_requirements: 512,
                precision: 0.01,
                coherence_requirements: 0.9,
                temperature_constraints: None,
                pressure_constraints: None,
                quantum_effects: false,
                consciousness_processing: false,
            },
            preferred_processors: vec![ProcessorType::GasOscillation],
            priority: 3,
            max_execution_time: Duration::from_secs(5),
            energy_budget: 500.0,
        };
        
        let energy = manager.calculate_energy_consumption(&task, ProcessorType::GasOscillation).await.unwrap();
        assert!(energy > 0.0);
    }

    #[test]
    async fn test_coherence_calculation() {
        let config = IntegrationConfig::default();
        let manager = IntegrationManager::new(config).unwrap();
        
        let task = UnifiedTask {
            id: Uuid::new_v4(),
            task_type: UnifiedTaskType::ConsciousnessProcessing,
            input_data: vec![1.0, 2.0, 3.0],
            requirements: ProcessingRequirements {
                computational_complexity: ComplexityLevel::ConsciousnessLevel,
                memory_requirements: 2048,
                precision: 0.0001,
                coherence_requirements: 0.99,
                temperature_constraints: None,
                pressure_constraints: None,
                quantum_effects: true,
                consciousness_processing: true,
            },
            preferred_processors: vec![ProcessorType::Consciousness],
            priority: 10,
            max_execution_time: Duration::from_secs(30),
            energy_budget: 2000.0,
        };
        
        let coherence = manager.calculate_coherence(&task, ProcessorType::Consciousness).await.unwrap();
        assert!(coherence > 0.9);
    }

    #[test]
    fn test_config_validation() {
        let mut config = IntegrationConfig::default();
        
        // Test valid configuration
        assert!(IntegrationManager::validate_config(&config).is_ok());
        
        // Test invalid configuration (all processors disabled)
        config.quantum_integration = false;
        config.neural_integration = false;
        config.fuzzy_integration = false;
        config.molecular_integration = false;
        config.bmd_integration = false;
        config.foundry_integration = false;
        config.gas_oscillation_integration = false;
        
        assert!(IntegrationManager::validate_config(&config).is_err());
    }
} 