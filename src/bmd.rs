//! Biological Maxwell Demon (BMD) information catalysis services
//!
//! This module handles BMD information catalysis for the Buhera framework,
//! implementing Mizraji's biological Maxwell demon theory with multi-scale
//! BMD networks and thermodynamic amplification.

use crate::error::{VPOSError, VPOSResult};
use crate::fuzzy::FuzzyValue;
use crate::semantic::SemanticContent;
use crate::quantum::QuantumCoherence;

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// BMD catalyst system implementing information catalysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BMDCatalyst {
    /// Catalyst identifier
    pub id: String,
    /// Pattern recognition threshold
    pub pattern_threshold: f64,
    /// Information catalysis efficiency
    pub catalysis_efficiency: f64,
    /// Thermodynamic amplification factor
    pub amplification_factor: f64,
    /// Input information filters
    pub input_filters: Vec<InformationFilter>,
    /// Output information channels
    pub output_channels: Vec<InformationChannel>,
    /// Entropy reduction tracking
    pub entropy_reduction: f64,
    /// Pattern recognition cache
    pub pattern_cache: HashMap<String, RecognizedPattern>,
    /// Catalysis history
    pub catalysis_history: Vec<CatalysisEvent>,
}

/// Information filter for BMD input processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationFilter {
    /// Filter identifier
    pub filter_id: String,
    /// Filter type
    pub filter_type: FilterType,
    /// Filter parameters
    pub parameters: HashMap<String, f64>,
    /// Selectivity coefficient
    pub selectivity: f64,
    /// Pattern recognition accuracy
    pub accuracy: f64,
}

/// Information channel for BMD output processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationChannel {
    /// Channel identifier
    pub channel_id: String,
    /// Channel type
    pub channel_type: ChannelType,
    /// Channel capacity
    pub capacity: f64,
    /// Output targeting precision
    pub precision: f64,
    /// Information throughput
    pub throughput: f64,
}

/// Filter type for information processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    /// Pattern recognition filter
    PatternRecognition,
    /// Frequency domain filter
    FrequencyDomain,
    /// Semantic content filter
    SemanticContent,
    /// Quantum coherence filter
    QuantumCoherence,
    /// Fuzzy logic filter
    FuzzyLogic,
}

/// Channel type for information output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelType {
    /// Direct output channel
    Direct,
    /// Amplified output channel
    Amplified,
    /// Coherent output channel
    Coherent,
    /// Semantic output channel
    Semantic,
    /// Quantum entangled channel
    QuantumEntangled,
}

/// Recognized pattern from BMD analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizedPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Pattern confidence
    pub confidence: f64,
    /// Pattern features
    pub features: Vec<f64>,
    /// Pattern classification
    pub classification: String,
    /// Recognition timestamp
    pub timestamp: Instant,
    /// Pattern entropy
    pub entropy: f64,
}

/// Catalysis event in BMD processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalysisEvent {
    /// Event identifier
    pub event_id: String,
    /// Input information entropy
    pub input_entropy: f64,
    /// Output information entropy
    pub output_entropy: f64,
    /// Entropy reduction achieved
    pub entropy_reduction: f64,
    /// Amplification factor applied
    pub amplification_factor: f64,
    /// Information preserved
    pub information_preserved: f64,
    /// Event timestamp
    pub timestamp: Instant,
}

/// BMD network for multi-scale coordination
#[derive(Debug, Clone)]
pub struct BMDNetwork {
    /// Network identifier
    pub network_id: String,
    /// BMD catalysts in network
    pub catalysts: Vec<BMDCatalyst>,
    /// Network topology
    pub topology: NetworkTopology,
    /// Coordination matrix
    pub coordination_matrix: CoordinationMatrix,
    /// Network coherence
    pub network_coherence: f64,
    /// Total amplification
    pub total_amplification: f64,
}

/// Network topology for BMD coordination
#[derive(Debug, Clone)]
pub struct NetworkTopology {
    /// Adjacency matrix
    pub adjacency_matrix: Vec<Vec<f64>>,
    /// Connection weights
    pub connection_weights: HashMap<String, f64>,
    /// Network diameter
    pub diameter: usize,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
}

/// Coordination matrix for BMD network
#[derive(Debug, Clone)]
pub struct CoordinationMatrix {
    /// Catalyst-to-catalyst coupling
    pub catalyst_coupling: Vec<Vec<f64>>,
    /// Information flow matrix
    pub information_flow: Vec<Vec<f64>>,
    /// Amplification coupling
    pub amplification_coupling: Vec<Vec<f64>>,
    /// Coherence coupling
    pub coherence_coupling: Vec<Vec<f64>>,
}

/// Information catalysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationCatalysisResult {
    /// Input information content
    pub input_information: Vec<f64>,
    /// Output information content
    pub output_information: Vec<f64>,
    /// Entropy reduction achieved
    pub entropy_reduction: f64,
    /// Information gain
    pub information_gain: f64,
    /// Amplification factor
    pub amplification_factor: f64,
    /// Catalysis efficiency
    pub catalysis_efficiency: f64,
    /// Processing time
    pub processing_time: Duration,
}

/// BMD chaos-to-order processor
#[derive(Debug, Clone)]
pub struct ChaosToOrderProcessor {
    /// Chaos detection threshold
    pub chaos_threshold: f64,
    /// Order generation parameters
    pub order_parameters: HashMap<String, f64>,
    /// Pattern emergence tracking
    pub pattern_emergence: Vec<EmergentPattern>,
    /// Complexity reduction metrics
    pub complexity_reduction: f64,
}

/// Emergent pattern from chaos processing
#[derive(Debug, Clone)]
pub struct EmergentPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Emergence probability
    pub emergence_probability: f64,
    /// Pattern stability
    pub stability: f64,
    /// Order parameter
    pub order_parameter: f64,
    /// Coherence length
    pub coherence_length: f64,
}

impl BMDCatalyst {
    /// Create a new BMD catalyst with specified parameters
    pub fn new(id: &str, catalysis_efficiency: f64) -> Self {
        Self {
            id: id.to_string(),
            pattern_threshold: 0.8,
            catalysis_efficiency,
            amplification_factor: 1.0,
            input_filters: vec![],
            output_channels: vec![],
            entropy_reduction: 0.0,
            pattern_cache: HashMap::new(),
            catalysis_history: vec![],
        }
    }

    /// Add information filter to catalyst
    pub fn add_input_filter(&mut self, filter: InformationFilter) {
        self.input_filters.push(filter);
    }

    /// Add output channel to catalyst
    pub fn add_output_channel(&mut self, channel: InformationChannel) {
        self.output_channels.push(channel);
    }

    /// Execute information catalysis (iCat = ℑinput ◦ ℑoutput)
    pub async fn execute_catalysis(
        &mut self,
        input_information: Vec<f64>,
    ) -> VPOSResult<InformationCatalysisResult> {
        let start_time = Instant::now();

        // Apply input filters (ℑinput)
        let filtered_input = self.apply_input_filters(&input_information).await?;

        // Pattern recognition and analysis
        let recognized_patterns = self.recognize_patterns(&filtered_input).await?;

        // Information catalysis core process
        let catalyzed_information = self.catalyze_information(&filtered_input, &recognized_patterns).await?;

        // Apply output channels (ℑoutput)
        let channeled_output = self.apply_output_channels(&catalyzed_information).await?;

        // Calculate entropy reduction
        let input_entropy = self.calculate_entropy(&input_information);
        let output_entropy = self.calculate_entropy(&channeled_output);
        let entropy_reduction = input_entropy - output_entropy;

        // Calculate amplification factor
        let amplification_factor = self.calculate_amplification_factor(&channeled_output, &input_information);

        // Record catalysis event
        let catalysis_event = CatalysisEvent {
            event_id: format!("{}_{}", self.id, chrono::Utc::now().timestamp()),
            input_entropy,
            output_entropy,
            entropy_reduction,
            amplification_factor,
            information_preserved: self.calculate_information_preservation(&input_information, &channeled_output),
            timestamp: Instant::now(),
        };
        self.catalysis_history.push(catalysis_event);

        // Update metrics
        self.entropy_reduction += entropy_reduction;
        self.amplification_factor = amplification_factor;

        Ok(InformationCatalysisResult {
            input_information,
            output_information: channeled_output,
            entropy_reduction,
            information_gain: entropy_reduction,
            amplification_factor,
            catalysis_efficiency: self.catalysis_efficiency,
            processing_time: start_time.elapsed(),
        })
    }

    /// Apply input information filters
    async fn apply_input_filters(&self, input: &[f64]) -> VPOSResult<Vec<f64>> {
        let mut filtered_input = input.to_vec();

        for filter in &self.input_filters {
            filtered_input = match filter.filter_type {
                FilterType::PatternRecognition => {
                    self.apply_pattern_recognition_filter(&filtered_input, filter).await?
                }
                FilterType::FrequencyDomain => {
                    self.apply_frequency_domain_filter(&filtered_input, filter).await?
                }
                FilterType::SemanticContent => {
                    self.apply_semantic_content_filter(&filtered_input, filter).await?
                }
                FilterType::QuantumCoherence => {
                    self.apply_quantum_coherence_filter(&filtered_input, filter).await?
                }
                FilterType::FuzzyLogic => {
                    self.apply_fuzzy_logic_filter(&filtered_input, filter).await?
                }
            };
        }

        Ok(filtered_input)
    }

    /// Apply output information channels
    async fn apply_output_channels(&self, output: &[f64]) -> VPOSResult<Vec<f64>> {
        let mut channeled_output = output.to_vec();

        for channel in &self.output_channels {
            channeled_output = match channel.channel_type {
                ChannelType::Direct => channeled_output,
                ChannelType::Amplified => {
                    self.apply_amplified_channel(&channeled_output, channel).await?
                }
                ChannelType::Coherent => {
                    self.apply_coherent_channel(&channeled_output, channel).await?
                }
                ChannelType::Semantic => {
                    self.apply_semantic_channel(&channeled_output, channel).await?
                }
                ChannelType::QuantumEntangled => {
                    self.apply_quantum_entangled_channel(&channeled_output, channel).await?
                }
            };
        }

        Ok(channeled_output)
    }

    /// Recognize patterns in filtered input
    async fn recognize_patterns(&mut self, input: &[f64]) -> VPOSResult<Vec<RecognizedPattern>> {
        let mut patterns = Vec::new();

        // Pattern recognition implementation
        for (i, &value) in input.iter().enumerate() {
            if value > self.pattern_threshold {
                let pattern_id = format!("pattern_{}_{}", self.id, i);
                let pattern = RecognizedPattern {
                    pattern_id: pattern_id.clone(),
                    confidence: value,
                    features: vec![value],
                    classification: self.classify_pattern(value),
                    timestamp: Instant::now(),
                    entropy: self.calculate_pattern_entropy(value),
                };

                // Cache the pattern
                self.pattern_cache.insert(pattern_id, pattern.clone());
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Catalyze information based on recognized patterns
    async fn catalyze_information(
        &self,
        input: &[f64],
        patterns: &[RecognizedPattern],
    ) -> VPOSResult<Vec<f64>> {
        let mut catalyzed = input.to_vec();

        for pattern in patterns {
            // Apply information catalysis based on pattern
            for (i, value) in catalyzed.iter_mut().enumerate() {
                if i < pattern.features.len() {
                    *value *= pattern.confidence * self.catalysis_efficiency;
                }
            }
        }

        Ok(catalyzed)
    }

    /// Recognize patterns in input data
    pub fn recognize_patterns(&self, input: &[f64]) -> VPOSResult<Vec<RecognizedPattern>> {
        let mut patterns = Vec::new();

        for (i, &value) in input.iter().enumerate() {
            if value > self.pattern_threshold {
                let pattern = RecognizedPattern {
                    pattern_id: format!("pattern_{}_{}", self.id, i),
                    confidence: value,
                    features: vec![value],
                    classification: self.classify_pattern(value),
                    timestamp: Instant::now(),
                    entropy: self.calculate_pattern_entropy(value),
                };
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Calculate entropy of information
    fn calculate_entropy(&self, information: &[f64]) -> f64 {
        let mut entropy = 0.0;
        let sum: f64 = information.iter().sum();
        
        if sum > 0.0 {
            for &value in information {
                if value > 0.0 {
                    let p = value / sum;
                    entropy -= p * p.log2();
                }
            }
        }
        
        entropy
    }

    /// Calculate amplification factor
    fn calculate_amplification_factor(&self, output: &[f64], input: &[f64]) -> f64 {
        let output_magnitude: f64 = output.iter().map(|x| x.abs()).sum();
        let input_magnitude: f64 = input.iter().map(|x| x.abs()).sum();
        
        if input_magnitude > 0.0 {
            output_magnitude / input_magnitude
        } else {
            1.0
        }
    }

    /// Calculate information preservation
    fn calculate_information_preservation(&self, input: &[f64], output: &[f64]) -> f64 {
        // Calculate correlation between input and output
        let input_mean: f64 = input.iter().sum::<f64>() / input.len() as f64;
        let output_mean: f64 = output.iter().sum::<f64>() / output.len() as f64;
        
        let numerator: f64 = input.iter().zip(output.iter())
            .map(|(i, o)| (i - input_mean) * (o - output_mean))
            .sum();
        
        let input_var: f64 = input.iter().map(|i| (i - input_mean).powi(2)).sum();
        let output_var: f64 = output.iter().map(|o| (o - output_mean).powi(2)).sum();
        
        let denominator = (input_var * output_var).sqrt();
        
        if denominator > 0.0 {
            (numerator / denominator).abs()
        } else {
            0.0
        }
    }

    /// Classify pattern based on value
    fn classify_pattern(&self, value: f64) -> String {
        if value > 0.9 {
            "high_confidence".to_string()
        } else if value > 0.7 {
            "medium_confidence".to_string()
        } else {
            "low_confidence".to_string()
        }
    }

    /// Calculate pattern entropy
    fn calculate_pattern_entropy(&self, value: f64) -> f64 {
        if value > 0.0 && value < 1.0 {
            -(value * value.log2() + (1.0 - value) * (1.0 - value).log2())
        } else {
            0.0
        }
    }

    // Filter implementations
    async fn apply_pattern_recognition_filter(&self, input: &[f64], filter: &InformationFilter) -> VPOSResult<Vec<f64>> {
        let threshold = filter.parameters.get("threshold").unwrap_or(&0.5);
        Ok(input.iter().map(|&x| if x > *threshold { x } else { 0.0 }).collect())
    }

    async fn apply_frequency_domain_filter(&self, input: &[f64], filter: &InformationFilter) -> VPOSResult<Vec<f64>> {
        let cutoff = filter.parameters.get("cutoff").unwrap_or(&0.5);
        // Simplified frequency domain filtering
        Ok(input.iter().map(|&x| x * cutoff).collect())
    }

    async fn apply_semantic_content_filter(&self, input: &[f64], filter: &InformationFilter) -> VPOSResult<Vec<f64>> {
        let semantic_weight = filter.parameters.get("semantic_weight").unwrap_or(&1.0);
        Ok(input.iter().map(|&x| x * semantic_weight).collect())
    }

    async fn apply_quantum_coherence_filter(&self, input: &[f64], filter: &InformationFilter) -> VPOSResult<Vec<f64>> {
        let coherence_factor = filter.parameters.get("coherence_factor").unwrap_or(&1.0);
        Ok(input.iter().map(|&x| x * coherence_factor).collect())
    }

    async fn apply_fuzzy_logic_filter(&self, input: &[f64], filter: &InformationFilter) -> VPOSResult<Vec<f64>> {
        let fuzzy_factor = filter.parameters.get("fuzzy_factor").unwrap_or(&1.0);
        Ok(input.iter().map(|&x| x * fuzzy_factor).collect())
    }

    // Channel implementations
    async fn apply_amplified_channel(&self, output: &[f64], channel: &InformationChannel) -> VPOSResult<Vec<f64>> {
        let amplification = channel.capacity;
        Ok(output.iter().map(|&x| x * amplification).collect())
    }

    async fn apply_coherent_channel(&self, output: &[f64], channel: &InformationChannel) -> VPOSResult<Vec<f64>> {
        let coherence = channel.precision;
        Ok(output.iter().map(|&x| x * coherence).collect())
    }

    async fn apply_semantic_channel(&self, output: &[f64], channel: &InformationChannel) -> VPOSResult<Vec<f64>> {
        let semantic_factor = channel.throughput;
        Ok(output.iter().map(|&x| x * semantic_factor).collect())
    }

    async fn apply_quantum_entangled_channel(&self, output: &[f64], channel: &InformationChannel) -> VPOSResult<Vec<f64>> {
        let entanglement_factor = channel.capacity * channel.precision;
        Ok(output.iter().map(|&x| x * entanglement_factor).collect())
    }
}

impl BMDNetwork {
    /// Create new BMD network
    pub fn new(network_id: &str) -> Self {
        Self {
            network_id: network_id.to_string(),
            catalysts: vec![],
            topology: NetworkTopology::new(),
            coordination_matrix: CoordinationMatrix::new(),
            network_coherence: 0.0,
            total_amplification: 1.0,
        }
    }

    /// Add catalyst to network
    pub fn add_catalyst(&mut self, catalyst: BMDCatalyst) {
        self.catalysts.push(catalyst);
        self.update_topology();
    }

    /// Execute coordinated BMD analysis
    pub async fn execute_coordinated_analysis(&mut self, input: Vec<f64>) -> VPOSResult<Vec<InformationCatalysisResult>> {
        let mut results = Vec::new();

        for catalyst in &mut self.catalysts {
            let result = catalyst.execute_catalysis(input.clone()).await?;
            results.push(result);
        }

        // Apply network coordination
        self.coordinate_results(&mut results).await?;

        Ok(results)
    }

    /// Coordinate results across network
    async fn coordinate_results(&mut self, results: &mut [InformationCatalysisResult]) -> VPOSResult<()> {
        // Calculate network amplification
        let total_amplification: f64 = results.iter().map(|r| r.amplification_factor).sum();
        self.total_amplification = total_amplification;

        // Update network coherence
        self.network_coherence = self.calculate_network_coherence(results);

        Ok(())
    }

    /// Calculate network coherence
    fn calculate_network_coherence(&self, results: &[InformationCatalysisResult]) -> f64 {
        let avg_entropy_reduction: f64 = results.iter().map(|r| r.entropy_reduction).sum::<f64>() / results.len() as f64;
        let avg_amplification: f64 = results.iter().map(|r| r.amplification_factor).sum::<f64>() / results.len() as f64;
        
        avg_entropy_reduction * avg_amplification / (avg_entropy_reduction + avg_amplification)
    }

    /// Update network topology
    fn update_topology(&mut self) {
        let n = self.catalysts.len();
        self.topology.adjacency_matrix = vec![vec![0.0; n]; n];
        
        // Create fully connected network for simplicity
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    self.topology.adjacency_matrix[i][j] = 1.0;
                }
            }
        }
    }
}

impl NetworkTopology {
    /// Create new network topology
    pub fn new() -> Self {
        Self {
            adjacency_matrix: vec![],
            connection_weights: HashMap::new(),
            diameter: 0,
            clustering_coefficient: 0.0,
        }
    }
}

impl CoordinationMatrix {
    /// Create new coordination matrix
    pub fn new() -> Self {
        Self {
            catalyst_coupling: vec![],
            information_flow: vec![],
            amplification_coupling: vec![],
            coherence_coupling: vec![],
        }
    }
}

impl ChaosToOrderProcessor {
    /// Create new chaos-to-order processor
    pub fn new() -> Self {
        Self {
            chaos_threshold: 0.5,
            order_parameters: HashMap::new(),
            pattern_emergence: vec![],
            complexity_reduction: 0.0,
        }
    }

    /// Process chaos into ordered patterns
    pub async fn process_chaos_to_order(&mut self, chaotic_input: Vec<f64>) -> VPOSResult<Vec<f64>> {
        let chaos_level = self.calculate_chaos_level(&chaotic_input);
        
        if chaos_level > self.chaos_threshold {
            let ordered_output = self.extract_order_from_chaos(&chaotic_input).await?;
            self.track_pattern_emergence(&chaotic_input, &ordered_output).await?;
            Ok(ordered_output)
        } else {
            Ok(chaotic_input)
        }
    }

    /// Calculate chaos level in input
    fn calculate_chaos_level(&self, input: &[f64]) -> f64 {
        let mean = input.iter().sum::<f64>() / input.len() as f64;
        let variance = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / input.len() as f64;
        variance.sqrt() / (mean.abs() + 1e-10)
    }

    /// Extract order from chaotic input
    async fn extract_order_from_chaos(&self, input: &[f64]) -> VPOSResult<Vec<f64>> {
        // Simple ordering by sorting and applying smoothing
        let mut ordered = input.to_vec();
        ordered.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Apply smoothing filter
        for i in 1..ordered.len()-1 {
            ordered[i] = (ordered[i-1] + ordered[i] + ordered[i+1]) / 3.0;
        }
        
        Ok(ordered)
    }

    /// Track pattern emergence
    async fn track_pattern_emergence(&mut self, input: &[f64], output: &[f64]) -> VPOSResult<()> {
        let emergence_probability = self.calculate_emergence_probability(input, output);
        let stability = self.calculate_pattern_stability(output);
        
        let pattern = EmergentPattern {
            pattern_id: format!("emergent_{}", chrono::Utc::now().timestamp()),
            emergence_probability,
            stability,
            order_parameter: self.calculate_order_parameter(output),
            coherence_length: self.calculate_coherence_length(output),
        };
        
        self.pattern_emergence.push(pattern);
        Ok(())
    }

    /// Calculate emergence probability
    fn calculate_emergence_probability(&self, input: &[f64], output: &[f64]) -> f64 {
        let input_entropy = self.calculate_entropy(input);
        let output_entropy = self.calculate_entropy(output);
        
        if input_entropy > 0.0 {
            1.0 - (output_entropy / input_entropy)
        } else {
            0.0
        }
    }

    /// Calculate pattern stability
    fn calculate_pattern_stability(&self, output: &[f64]) -> f64 {
        let mean = output.iter().sum::<f64>() / output.len() as f64;
        let variance = output.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / output.len() as f64;
        
        1.0 / (1.0 + variance)
    }

    /// Calculate order parameter
    fn calculate_order_parameter(&self, output: &[f64]) -> f64 {
        let sorted_output = {
            let mut sorted = output.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted
        };
        
        let correlation = self.calculate_correlation(output, &sorted_output);
        correlation.abs()
    }

    /// Calculate coherence length
    fn calculate_coherence_length(&self, output: &[f64]) -> f64 {
        let mut coherence_length = 0.0;
        let threshold = 0.1;
        
        for i in 0..output.len()-1 {
            let diff = (output[i] - output[i+1]).abs();
            if diff < threshold {
                coherence_length += 1.0;
            } else {
                break;
            }
        }
        
        coherence_length
    }

    /// Calculate entropy
    fn calculate_entropy(&self, data: &[f64]) -> f64 {
        let mut entropy = 0.0;
        let sum: f64 = data.iter().map(|x| x.abs()).sum();
        
        if sum > 0.0 {
            for &value in data {
                if value.abs() > 0.0 {
                    let p = value.abs() / sum;
                    entropy -= p * p.log2();
                }
            }
        }
        
        entropy
    }

    /// Calculate correlation
    fn calculate_correlation(&self, a: &[f64], b: &[f64]) -> f64 {
        let mean_a = a.iter().sum::<f64>() / a.len() as f64;
        let mean_b = b.iter().sum::<f64>() / b.len() as f64;
        
        let numerator: f64 = a.iter().zip(b.iter())
            .map(|(x, y)| (x - mean_a) * (y - mean_b))
            .sum();
        
        let var_a: f64 = a.iter().map(|x| (x - mean_a).powi(2)).sum();
        let var_b: f64 = b.iter().map(|y| (y - mean_b).powi(2)).sum();
        
        let denominator = (var_a * var_b).sqrt();
        
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

impl Default for BMDCatalyst {
    fn default() -> Self {
        Self::new("default", 0.8)
    }
}

impl Default for BMDNetwork {
    fn default() -> Self {
        Self::new("default_network")
    }
}

impl Default for ChaosToOrderProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for NetworkTopology {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CoordinationMatrix {
    fn default() -> Self {
        Self::new()
    }
} 