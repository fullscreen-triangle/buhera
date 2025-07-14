//! Fuzzy digital state management and computation
//!
//! This module implements fuzzy logic and continuous-valued computation for the Buhera framework.
//! Transcends binary limitations through continuous-valued gate states and gradual transitions.

use crate::error::{BuheraResult, BuheraError};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Fuzzy value with membership function and confidence
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzyValue {
    /// Fuzzy data value in [0,1]
    pub value: f64,
    /// Membership function value in [0,1]
    pub membership: f64,
    /// Confidence in the stored value in [0,1]
    pub confidence: f64,
}

impl FuzzyValue {
    /// Create new fuzzy value
    pub fn new(value: f64, membership: f64, confidence: f64) -> BuheraResult<Self> {
        if value < 0.0 || value > 1.0 || membership < 0.0 || membership > 1.0 || confidence < 0.0 || confidence > 1.0 {
            return Err(BuheraError::InvalidInput("Fuzzy values must be in [0,1] range".to_string()));
        }
        Ok(Self { value, membership, confidence })
    }

    /// Fuzzy AND operation
    pub fn and(&self, other: &FuzzyValue) -> FuzzyValue {
        FuzzyValue {
            value: self.value.min(other.value),
            membership: self.membership.min(other.membership),
            confidence: (self.confidence * other.confidence).sqrt(),
        }
    }

    /// Fuzzy OR operation  
    pub fn or(&self, other: &FuzzyValue) -> FuzzyValue {
        FuzzyValue {
            value: self.value.max(other.value),
            membership: self.membership.max(other.membership),
            confidence: (self.confidence * other.confidence).sqrt(),
        }
    }

    /// Fuzzy NOT operation
    pub fn not(&self) -> FuzzyValue {
        FuzzyValue {
            value: 1.0 - self.value,
            membership: 1.0 - self.membership,
            confidence: self.confidence,
        }
    }

    /// Defuzzify to crisp value
    pub fn defuzzify(&self) -> f64 {
        self.value * self.membership * self.confidence
    }
}

/// Fuzzy gate with continuous state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyGate {
    /// Gate identifier
    pub id: String,
    /// Current gate state in [0,1]
    pub state: f64,
    /// Gate conductance
    pub conductance: f64,
    /// Input history for context-dependent processing
    pub input_history: Vec<FuzzyValue>,
    /// Process context influence
    pub process_context: HashMap<String, f64>,
    /// Transition parameters
    pub transition_params: FuzzyTransitionParams,
}

/// Fuzzy transition parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyTransitionParams {
    /// Transition alpha (input strength)
    pub alpha: f64,
    /// Transition beta (state decay)
    pub beta: f64,
    /// Transition gamma (context influence)
    pub gamma: f64,
    /// Transition time constant
    pub time_constant: f64,
}

impl FuzzyGate {
    /// Create new fuzzy gate
    pub fn new(id: String) -> Self {
        Self {
            id,
            state: 0.5,
            conductance: 1.0,
            input_history: Vec::new(),
            process_context: HashMap::new(),
            transition_params: FuzzyTransitionParams {
                alpha: 0.8,
                beta: 0.1,
                gamma: 0.3,
                time_constant: 1.0,
            },
        }
    }

    /// Process fuzzy input with gradual transition
    pub fn process_input(&mut self, input: FuzzyValue, dt: f64) -> BuheraResult<FuzzyValue> {
        // Update input history
        self.input_history.push(input.clone());
        if self.input_history.len() > 100 {
            self.input_history.remove(0);
        }

        // Calculate context influence
        let context_influence: f64 = self.process_context.values().sum::<f64>() / self.process_context.len() as f64;

        // Apply gradual transition dynamics
        let input_strength = input.value * input.membership * input.confidence;
        let state_decay = self.state * self.transition_params.beta;
        let context_effect = context_influence * self.transition_params.gamma;

        // State evolution equation
        let d_state = (self.transition_params.alpha * input_strength - state_decay + context_effect) * dt;
        self.state = (self.state + d_state).clamp(0.0, 1.0);

        // Update conductance based on state
        self.conductance = self.state;

        // Generate output with process-dependent behavior
        let output_value = self.state * self.conductance;
        let output_membership = (input.membership + self.state) / 2.0;
        let output_confidence = input.confidence * self.state;

        Ok(FuzzyValue {
            value: output_value,
            membership: output_membership,
            confidence: output_confidence,
        })
    }

    /// Update process context
    pub fn update_context(&mut self, key: String, value: f64) {
        self.process_context.insert(key, value.clamp(0.0, 1.0));
    }

    /// Get gate state
    pub fn get_state(&self) -> f64 {
        self.state
    }
}

/// Fuzzy memory with continuous addressing
#[derive(Debug, Clone)]
pub struct FuzzyMemory {
    /// Memory identifier
    pub id: String,
    /// Memory data with fuzzy addressing
    pub data: HashMap<String, FuzzyValue>,
    /// Memory capacity
    pub capacity: usize,
    /// Access history
    pub access_history: Vec<FuzzyMemoryAccess>,
}

/// Fuzzy memory access record
#[derive(Debug, Clone)]
pub struct FuzzyMemoryAccess {
    /// Address accessed
    pub address: String,
    /// Access type
    pub access_type: FuzzyAccessType,
    /// Timestamp
    pub timestamp: Instant,
    /// Value accessed
    pub value: FuzzyValue,
}

/// Fuzzy memory access type
#[derive(Debug, Clone)]
pub enum FuzzyAccessType {
    Read,
    Write,
    Update,
}

impl FuzzyMemory {
    /// Create new fuzzy memory
    pub fn new(id: String, capacity: usize) -> Self {
        Self {
            id,
            data: HashMap::new(),
            capacity,
            access_history: Vec::new(),
        }
    }

    /// Read fuzzy value from memory
    pub fn read(&mut self, address: &str) -> BuheraResult<FuzzyValue> {
        let value = self.data.get(address).cloned().unwrap_or_else(|| {
            FuzzyValue { value: 0.0, membership: 0.0, confidence: 0.0 }
        });

        self.access_history.push(FuzzyMemoryAccess {
            address: address.to_string(),
            access_type: FuzzyAccessType::Read,
            timestamp: Instant::now(),
            value: value.clone(),
        });

        Ok(value)
    }

    /// Write fuzzy value to memory
    pub fn write(&mut self, address: &str, value: FuzzyValue) -> BuheraResult<()> {
        if self.data.len() >= self.capacity && !self.data.contains_key(address) {
            return Err(BuheraError::MemoryError("Fuzzy memory capacity exceeded".to_string()));
        }

        self.data.insert(address.to_string(), value.clone());

        self.access_history.push(FuzzyMemoryAccess {
            address: address.to_string(),
            access_type: FuzzyAccessType::Write,
            timestamp: Instant::now(),
            value,
        });

        Ok(())
    }

    /// Update fuzzy value in memory
    pub fn update(&mut self, address: &str, updater: impl Fn(&FuzzyValue) -> FuzzyValue) -> BuheraResult<()> {
        if let Some(current_value) = self.data.get(address) {
            let updated_value = updater(current_value);
            self.data.insert(address.to_string(), updated_value.clone());

            self.access_history.push(FuzzyMemoryAccess {
                address: address.to_string(),
                access_type: FuzzyAccessType::Update,
                timestamp: Instant::now(),
                value: updated_value,
            });

            Ok(())
        } else {
            Err(BuheraError::MemoryError(format!("Address {} not found in fuzzy memory", address)))
        }
    }
}

/// Fuzzy state manager with full continuous-valued computation
#[derive(Debug)]
pub struct FuzzyStateManager {
    /// Fuzzy precision
    pub precision: f64,
    /// Fuzzy gates registry
    pub gates: HashMap<String, FuzzyGate>,
    /// Fuzzy memory units
    pub memory_units: HashMap<String, FuzzyMemory>,
    /// Process context
    pub global_context: HashMap<String, f64>,
    /// System state
    pub system_state: FuzzySystemState,
}

/// Fuzzy system state
#[derive(Debug, Clone)]
pub struct FuzzySystemState {
    /// Average gate state
    pub average_gate_state: f64,
    /// Total memory usage
    pub memory_usage: f64,
    /// System coherence
    pub coherence: f64,
    /// Error rate
    pub error_rate: f64,
}

impl FuzzyStateManager {
    /// Create a new fuzzy state manager
    pub fn new() -> Self {
        Self {
            precision: 0.001,
            gates: HashMap::new(),
            memory_units: HashMap::new(),
            global_context: HashMap::new(),
            system_state: FuzzySystemState {
                average_gate_state: 0.5,
                memory_usage: 0.0,
                coherence: 1.0,
                error_rate: 0.0,
            },
        }
    }

    /// Add fuzzy gate to system
    pub fn add_gate(&mut self, gate: FuzzyGate) {
        self.gates.insert(gate.id.clone(), gate);
    }

    /// Add fuzzy memory unit
    pub fn add_memory_unit(&mut self, memory: FuzzyMemory) {
        self.memory_units.insert(memory.id.clone(), memory);
    }

    /// Process fuzzy computation across all gates
    pub fn process_computation(&mut self, inputs: HashMap<String, FuzzyValue>, dt: f64) -> BuheraResult<HashMap<String, FuzzyValue>> {
        let mut outputs = HashMap::new();

        for (gate_id, input) in inputs {
            if let Some(gate) = self.gates.get_mut(&gate_id) {
                let output = gate.process_input(input, dt)?;
                outputs.insert(gate_id, output);
            }
        }

        // Update system state
        self.update_system_state();

        Ok(outputs)
    }

    /// Update global context
    pub fn update_global_context(&mut self, key: String, value: f64) {
        self.global_context.insert(key, value.clamp(0.0, 1.0));
        
        // Propagate to all gates
        for gate in self.gates.values_mut() {
            gate.update_context(key.clone(), value);
        }
    }

    /// Update system state
    fn update_system_state(&mut self) {
        // Calculate average gate state
        let gate_states: Vec<f64> = self.gates.values().map(|g| g.state).collect();
        self.system_state.average_gate_state = gate_states.iter().sum::<f64>() / gate_states.len() as f64;

        // Calculate memory usage
        let total_memory: usize = self.memory_units.values().map(|m| m.data.len()).sum();
        let total_capacity: usize = self.memory_units.values().map(|m| m.capacity).sum();
        self.system_state.memory_usage = if total_capacity > 0 {
            total_memory as f64 / total_capacity as f64
        } else {
            0.0
        };

        // Calculate coherence based on gate state variance
        let variance = gate_states.iter()
            .map(|&state| (state - self.system_state.average_gate_state).powi(2))
            .sum::<f64>() / gate_states.len() as f64;
        self.system_state.coherence = (1.0 - variance).max(0.0);

        // Simple error rate calculation
        self.system_state.error_rate = variance;
    }

    /// Get system state
    pub fn get_system_state(&self) -> &FuzzySystemState {
        &self.system_state
    }

    /// Perform fuzzy inference
    pub fn fuzzy_inference(&self, input: FuzzyValue, rules: &[(FuzzyValue, FuzzyValue)]) -> FuzzyValue {
        let mut output_value = 0.0;
        let mut output_membership = 0.0;
        let mut output_confidence = 0.0;
        let mut total_weight = 0.0;

        for (antecedent, consequent) in rules {
            // Calculate rule activation
            let activation = input.and(antecedent);
            let weight = activation.defuzzify();

            if weight > 0.0 {
                output_value += weight * consequent.value;
                output_membership += weight * consequent.membership;
                output_confidence += weight * consequent.confidence;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            FuzzyValue {
                value: output_value / total_weight,
                membership: output_membership / total_weight,
                confidence: output_confidence / total_weight,
            }
        } else {
            FuzzyValue { value: 0.0, membership: 0.0, confidence: 0.0 }
        }
    }
}

impl Default for FuzzyStateManager {
    fn default() -> Self {
        Self::new()
    }
} 