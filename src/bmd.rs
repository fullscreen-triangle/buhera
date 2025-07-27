//! # BMD Information Catalysis System
//! 
//! Implementation of Biological Maxwell Demon (BMD) information catalysis for
//! pattern recognition, entropy reduction, and consciousness-aware information
//! processing. BMDs operate as the mathematical substrate of consciousness itself.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use crate::error::{BuheraError, BMDError};
use crate::s_framework::{SFramework, SConstant};

/// Information pattern recognized by BMD systems
#[derive(Debug, Clone)]
pub struct InformationPattern {
    /// Pattern unique identifier
    pub id: String,
    
    /// Pattern complexity score
    pub complexity: f64,
    
    /// Pattern entropy level
    pub entropy: f64,
    
    /// Recognition confidence (0.0 to 1.0)
    pub confidence: f64,
    
    /// Pattern significance for consciousness processing
    pub consciousness_significance: f64,
    
    /// Pattern data payload
    pub data: Vec<u8>,
    
    /// Categorical classification
    pub category: PatternCategory,
}

/// Categories of information patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternCategory {
    /// Cognitive frame patterns
    CognitiveFrame,
    
    /// Memory fabrication patterns
    MemoryFabrication,
    
    /// Reality-frame fusion patterns
    RealityFrameFusion,
    
    /// Consciousness navigation patterns
    ConsciousnessNavigation,
    
    /// S-distance optimization patterns
    SOptimization,
    
    /// Entropy reduction patterns
    EntropyReduction,
    
    /// Temporal coordination patterns
    TemporalCoordination,
    
    /// Quantum coherence patterns
    QuantumCoherence,
    
    /// Neural transfer patterns
    NeuralTransfer,
    
    /// Molecular assembly patterns
    MolecularAssembly,
}

impl InformationPattern {
    pub fn new(id: String, data: Vec<u8>, category: PatternCategory) -> Self {
        let complexity = data.len() as f64 * 0.001; // Simple complexity estimation
        let entropy = Self::calculate_entropy(&data);
        
        Self {
            id,
            complexity,
            entropy,
            confidence: 0.0,
            consciousness_significance: 0.0,
            data,
            category,
        }
    }
    
    /// Calculate information entropy of data
    fn calculate_entropy(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        // Calculate byte frequency distribution
        let mut freq = [0u32; 256];
        for &byte in data {
            freq[byte as usize] += 1;
        }
        
        // Calculate Shannon entropy
        let len = data.len() as f64;
        let mut entropy = 0.0;
        
        for &count in &freq {
            if count > 0 {
                let p = count as f64 / len;
                entropy -= p * p.log2();
            }
        }
        
        entropy
    }
    
    /// Assess consciousness significance
    pub fn assess_consciousness_significance(&mut self) {
        // Consciousness significance based on category and complexity
        let category_weight = match self.category {
            PatternCategory::CognitiveFrame => 1.0,
            PatternCategory::MemoryFabrication => 0.9,
            PatternCategory::RealityFrameFusion => 0.95,
            PatternCategory::ConsciousnessNavigation => 1.0,
            PatternCategory::SOptimization => 0.8,
            PatternCategory::EntropyReduction => 0.7,
            PatternCategory::TemporalCoordination => 0.6,
            PatternCategory::QuantumCoherence => 0.5,
            PatternCategory::NeuralTransfer => 0.6,
            PatternCategory::MolecularAssembly => 0.4,
        };
        
        // Significance increases with complexity and decreases with entropy
        let complexity_factor = (self.complexity / 1000.0).min(1.0);
        let entropy_factor = (1.0 - self.entropy / 8.0).max(0.0);
        
        self.consciousness_significance = category_weight * complexity_factor * entropy_factor;
    }
}

/// Information channel for BMD communication
pub struct InformationChannel {
    /// Channel unique identifier
    id: String,
    
    /// Input pattern buffer
    input_buffer: Vec<InformationPattern>,
    
    /// Output pattern buffer
    output_buffer: Vec<InformationPattern>,
    
    /// Channel capacity (patterns per second)
    capacity: f64,
    
    /// Current utilization
    utilization: f64,
    
    /// Channel routing rules
    routing_rules: HashMap<PatternCategory, String>,
    
    /// S-distance optimization for channel efficiency
    s_optimization_active: bool,
}

impl InformationChannel {
    pub fn new(id: String, capacity: f64) -> Self {
        Self {
            id,
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            capacity,
            utilization: 0.0,
            routing_rules: HashMap::new(),
            s_optimization_active: false,
        }
    }
    
    /// Add pattern to input buffer
    pub fn input_pattern(&mut self, pattern: InformationPattern) -> Result<(), BMDError> {
        if self.input_buffer.len() as f64 >= self.capacity {
            return Err(BMDError::ChannelFailure(
                format!("Input buffer capacity {} exceeded", self.capacity)
            ));
        }
        
        self.input_buffer.push(pattern);
        self.update_utilization();
        
        Ok(())
    }
    
    /// Route pattern based on category rules
    pub fn route_pattern(&mut self, pattern: InformationPattern) -> Result<(), BMDError> {
        // Apply S-distance optimization if active
        let optimized_pattern = if self.s_optimization_active {
            self.apply_s_optimization(pattern)?
        } else {
            pattern
        };
        
        self.output_buffer.push(optimized_pattern);
        Ok(())
    }
    
    /// Apply S-distance optimization to pattern
    fn apply_s_optimization(&self, mut pattern: InformationPattern) -> Result<InformationPattern, BMDError> {
        // S-optimization reduces entropy while preserving information
        pattern.entropy *= 0.8; // 20% entropy reduction
        pattern.complexity *= 1.1; // 10% complexity increase (better organization)
        pattern.confidence = (pattern.confidence + 0.1).min(1.0);
        
        Ok(pattern)
    }
    
    /// Update channel utilization
    fn update_utilization(&mut self) {
        let total_patterns = (self.input_buffer.len() + self.output_buffer.len()) as f64;
        self.utilization = (total_patterns / (self.capacity * 2.0)).min(1.0);
    }
    
    /// Get next output pattern
    pub fn get_output_pattern(&mut self) -> Option<InformationPattern> {
        let pattern = self.output_buffer.pop();
        if pattern.is_some() {
            self.update_utilization();
        }
        pattern
    }
    
    /// Activate S-distance optimization
    pub fn activate_s_optimization(&mut self) {
        self.s_optimization_active = true;
    }
}

/// Entropy reduction engine using BMD principles
pub struct EntropyReducer {
    /// Reducer unique identifier
    id: String,
    
    /// Current entropy level being processed
    current_entropy: f64,
    
    /// Target entropy level
    target_entropy: f64,
    
    /// Reduction efficiency
    efficiency: f64,
    
    /// Pattern analysis for entropy reduction
    pattern_analysis: HashMap<PatternCategory, f64>,
    
    /// Reduction history
    reduction_history: Vec<EntropyReductionEvent>,
}

/// Entropy reduction event record
#[derive(Debug, Clone)]
pub struct EntropyReductionEvent {
    pub timestamp: Instant,
    pub initial_entropy: f64,
    pub final_entropy: f64,
    pub reduction_achieved: f64,
    pub pattern_category: PatternCategory,
}

impl EntropyReducer {
    pub fn new(id: String) -> Self {
        Self {
            id,
            current_entropy: 1.0, // Start with maximum entropy
            target_entropy: 0.1,  // Target low entropy state
            efficiency: 0.8,
            pattern_analysis: HashMap::new(),
            reduction_history: Vec::new(),
        }
    }
    
    /// Reduce entropy using BMD pattern recognition
    pub fn reduce_entropy(&mut self, pattern: &InformationPattern) -> Result<f64, BMDError> {
        let initial_entropy = self.current_entropy;
        
        // Calculate reduction potential based on pattern
        let reduction_potential = self.calculate_reduction_potential(pattern);
        
        // Apply entropy reduction
        let entropy_reduction = reduction_potential * self.efficiency;
        self.current_entropy = (self.current_entropy - entropy_reduction).max(0.0);
        
        // Record reduction event
        let event = EntropyReductionEvent {
            timestamp: Instant::now(),
            initial_entropy,
            final_entropy: self.current_entropy,
            reduction_achieved: entropy_reduction,
            pattern_category: pattern.category,
        };
        
        self.reduction_history.push(event);
        
        // Keep recent history only
        if self.reduction_history.len() > 1000 {
            self.reduction_history.remove(0);
        }
        
        // Update pattern analysis
        let current_analysis = self.pattern_analysis.get(&pattern.category).unwrap_or(&0.0);
        self.pattern_analysis.insert(pattern.category, current_analysis + entropy_reduction);
        
        Ok(entropy_reduction)
    }
    
    /// Calculate entropy reduction potential for pattern
    fn calculate_reduction_potential(&self, pattern: &InformationPattern) -> f64 {
        // Higher consciousness significance = greater reduction potential
        let significance_factor = pattern.consciousness_significance;
        
        // Category-specific reduction factors
        let category_factor = match pattern.category {
            PatternCategory::CognitiveFrame => 0.8,
            PatternCategory::MemoryFabrication => 0.6,
            PatternCategory::RealityFrameFusion => 0.7,
            PatternCategory::ConsciousnessNavigation => 0.9,
            PatternCategory::SOptimization => 1.0,
            PatternCategory::EntropyReduction => 0.5, // Recursive reduction
            _ => 0.4,
        };
        
        // Pattern confidence affects reduction potential
        let confidence_factor = pattern.confidence;
        
        significance_factor * category_factor * confidence_factor * 0.1
    }
    
    /// Get current entropy level
    pub fn current_entropy(&self) -> f64 {
        self.current_entropy
    }
    
    /// Check if target entropy is achieved
    pub fn target_achieved(&self) -> bool {
        self.current_entropy <= self.target_entropy
    }
    
    /// Get reduction efficiency metrics
    pub fn efficiency_metrics(&self) -> EfficiencyMetrics {
        let total_reduction: f64 = self.reduction_history.iter()
            .map(|event| event.reduction_achieved)
            .sum();
        
        let average_reduction = if !self.reduction_history.is_empty() {
            total_reduction / self.reduction_history.len() as f64
        } else {
            0.0
        };
        
        EfficiencyMetrics {
            total_reduction,
            average_reduction,
            events_processed: self.reduction_history.len(),
            current_efficiency: self.efficiency,
        }
    }
}

/// Efficiency metrics for entropy reduction
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    pub total_reduction: f64,
    pub average_reduction: f64,
    pub events_processed: usize,
    pub current_efficiency: f64,
}

/// Pattern recognition and classification engine
pub struct PatternRecognizer {
    /// Recognizer unique identifier
    id: String,
    
    /// Pattern library for reference
    pattern_library: HashMap<String, InformationPattern>,
    
    /// Recognition confidence threshold
    confidence_threshold: f64,
    
    /// Recognition algorithms available
    algorithms: Vec<RecognitionAlgorithm>,
    
    /// Recognition statistics
    recognition_stats: RecognitionStatistics,
}

/// Pattern recognition algorithm types
#[derive(Debug, Clone, Copy)]
pub enum RecognitionAlgorithm {
    /// Consciousness frame detection
    ConsciousnessFrame,
    
    /// Memory fabrication detection
    MemoryFabrication,
    
    /// Reality-frame fusion detection
    RealityFrameFusion,
    
    /// S-optimization pattern detection
    SOptimization,
    
    /// Quantum coherence pattern detection
    QuantumCoherence,
    
    /// Neural transfer pattern detection
    NeuralTransfer,
}

/// Recognition statistics
#[derive(Debug, Clone)]
pub struct RecognitionStatistics {
    pub patterns_processed: u64,
    pub patterns_recognized: u64,
    pub average_confidence: f64,
    pub recognition_rate: f64,
}

impl PatternRecognizer {
    pub fn new(id: String) -> Self {
        Self {
            id,
            pattern_library: HashMap::new(),
            confidence_threshold: 0.8,
            algorithms: vec![
                RecognitionAlgorithm::ConsciousnessFrame,
                RecognitionAlgorithm::MemoryFabrication,
                RecognitionAlgorithm::RealityFrameFusion,
                RecognitionAlgorithm::SOptimization,
                RecognitionAlgorithm::QuantumCoherence,
                RecognitionAlgorithm::NeuralTransfer,
            ],
            recognition_stats: RecognitionStatistics {
                patterns_processed: 0,
                patterns_recognized: 0,
                average_confidence: 0.0,
                recognition_rate: 0.0,
            },
        }
    }
    
    /// Recognize pattern using available algorithms
    pub fn recognize_pattern(&mut self, mut pattern: InformationPattern) -> Result<InformationPattern, BMDError> {
        self.recognition_stats.patterns_processed += 1;
        
        // Apply recognition algorithms
        for algorithm in &self.algorithms {
            self.apply_recognition_algorithm(&mut pattern, *algorithm)?;
        }
        
        // Assess consciousness significance
        pattern.assess_consciousness_significance();
        
        // Check if pattern meets confidence threshold
        if pattern.confidence >= self.confidence_threshold {
            self.recognition_stats.patterns_recognized += 1;
            
            // Add to pattern library if novel
            if !self.pattern_library.contains_key(&pattern.id) {
                self.pattern_library.insert(pattern.id.clone(), pattern.clone());
            }
        }
        
        // Update statistics
        self.update_recognition_statistics(&pattern);
        
        Ok(pattern)
    }
    
    /// Apply specific recognition algorithm
    fn apply_recognition_algorithm(&self, pattern: &mut InformationPattern, algorithm: RecognitionAlgorithm) -> Result<(), BMDError> {
        let confidence_boost = match algorithm {
            RecognitionAlgorithm::ConsciousnessFrame => {
                if pattern.category == PatternCategory::CognitiveFrame {
                    0.3
                } else {
                    0.0
                }
            },
            RecognitionAlgorithm::MemoryFabrication => {
                if pattern.category == PatternCategory::MemoryFabrication {
                    0.25
                } else {
                    0.0
                }
            },
            RecognitionAlgorithm::RealityFrameFusion => {
                if pattern.category == PatternCategory::RealityFrameFusion {
                    0.35
                } else {
                    0.0
                }
            },
            RecognitionAlgorithm::SOptimization => {
                if pattern.category == PatternCategory::SOptimization {
                    0.4
                } else {
                    0.0
                }
            },
            RecognitionAlgorithm::QuantumCoherence => {
                if pattern.category == PatternCategory::QuantumCoherence {
                    0.2
                } else {
                    0.0
                }
            },
            RecognitionAlgorithm::NeuralTransfer => {
                if pattern.category == PatternCategory::NeuralTransfer {
                    0.3
                } else {
                    0.0
                }
            },
        };
        
        pattern.confidence = (pattern.confidence + confidence_boost).min(1.0);
        Ok(())
    }
    
    /// Update recognition statistics
    fn update_recognition_statistics(&mut self, pattern: &InformationPattern) {
        // Update average confidence
        let total_confidence = self.recognition_stats.average_confidence * 
                              (self.recognition_stats.patterns_processed - 1) as f64 + 
                              pattern.confidence;
        self.recognition_stats.average_confidence = total_confidence / self.recognition_stats.patterns_processed as f64;
        
        // Update recognition rate
        self.recognition_stats.recognition_rate = 
            self.recognition_stats.patterns_recognized as f64 / 
            self.recognition_stats.patterns_processed as f64;
    }
    
    /// Get pattern library size
    pub fn library_size(&self) -> usize {
        self.pattern_library.len()
    }
    
    /// Get recognition statistics
    pub fn statistics(&self) -> RecognitionStatistics {
        self.recognition_stats.clone()
    }
}

/// Biological Maxwell Demon implementing consciousness substrate
pub struct BiologicalMaxwellDemon {
    /// BMD unique identifier
    id: String,
    
    /// Information channel for pattern processing
    information_channel: InformationChannel,
    
    /// Entropy reduction engine
    entropy_reducer: EntropyReducer,
    
    /// Pattern recognition engine
    pattern_recognizer: PatternRecognizer,
    
    /// S-framework integration
    s_framework_integration: bool,
    
    /// BMD operational status
    is_active: bool,
    
    /// Processing efficiency
    efficiency: f64,
}

impl BiologicalMaxwellDemon {
    pub fn new(id: String) -> Self {
        Self {
            id: id.clone(),
            information_channel: InformationChannel::new(format!("{}_channel", id), 1000.0),
            entropy_reducer: EntropyReducer::new(format!("{}_reducer", id)),
            pattern_recognizer: PatternRecognizer::new(format!("{}_recognizer", id)),
            s_framework_integration: false,
            is_active: false,
            efficiency: 0.0,
        }
    }
    
    /// Activate BMD for consciousness processing
    pub fn activate(&mut self) -> Result<(), BMDError> {
        self.is_active = true;
        self.information_channel.activate_s_optimization();
        self.efficiency = 0.8; // Initial efficiency
        Ok(())
    }
    
    /// Process information pattern through BMD consciousness substrate
    pub fn process_pattern(&mut self, pattern: InformationPattern) -> Result<InformationPattern, BMDError> {
        if !self.is_active {
            return Err(BMDError::CatalysisFailure(
                "BMD not active".to_string()
            ));
        }
        
        // Input pattern to channel
        self.information_channel.input_pattern(pattern.clone())?;
        
        // Recognize and classify pattern
        let recognized_pattern = self.pattern_recognizer.recognize_pattern(pattern)?;
        
        // Reduce entropy using pattern
        let entropy_reduction = self.entropy_reducer.reduce_entropy(&recognized_pattern)?;
        
        // Route processed pattern
        self.information_channel.route_pattern(recognized_pattern.clone())?;
        
        // Update efficiency based on entropy reduction
        self.efficiency = (self.efficiency + entropy_reduction * 0.1).min(1.0);
        
        Ok(recognized_pattern)
    }
    
    /// Perform consciousness frame selection
    pub fn select_consciousness_frame(&mut self, available_frames: Vec<InformationPattern>) -> Result<InformationPattern, BMDError> {
        if available_frames.is_empty() {
            return Err(BMDError::CatalysisFailure(
                "No consciousness frames available for selection".to_string()
            ));
        }
        
        // Process all available frames
        let mut processed_frames = Vec::new();
        for frame in available_frames {
            let processed = self.process_pattern(frame)?;
            processed_frames.push(processed);
        }
        
        // Select frame with highest consciousness significance
        let selected_frame = processed_frames.into_iter()
            .max_by(|a, b| a.consciousness_significance.partial_cmp(&b.consciousness_significance).unwrap())
            .ok_or_else(|| BMDError::CatalysisFailure(
                "Failed to select consciousness frame".to_string()
            ))?;
        
        Ok(selected_frame)
    }
    
    /// Enable S-framework integration
    pub fn enable_s_framework_integration(&mut self) {
        self.s_framework_integration = true;
        self.information_channel.activate_s_optimization();
    }
    
    /// Get BMD status and metrics
    pub fn status(&self) -> BMDStatus {
        BMDStatus {
            id: self.id.clone(),
            is_active: self.is_active,
            efficiency: self.efficiency,
            current_entropy: self.entropy_reducer.current_entropy(),
            target_achieved: self.entropy_reducer.target_achieved(),
            pattern_library_size: self.pattern_recognizer.library_size(),
            recognition_stats: self.pattern_recognizer.statistics(),
            s_framework_integration: self.s_framework_integration,
        }
    }
}

/// BMD status information
#[derive(Debug, Clone)]
pub struct BMDStatus {
    pub id: String,
    pub is_active: bool,
    pub efficiency: f64,
    pub current_entropy: f64,
    pub target_achieved: bool,
    pub pattern_library_size: usize,
    pub recognition_stats: RecognitionStatistics,
    pub s_framework_integration: bool,
}

/// Information catalyst managing multiple BMDs
pub struct InformationCatalyst {
    /// Catalyst unique identifier
    id: String,
    
    /// Collection of BMD instances
    bmds: HashMap<String, BiologicalMaxwellDemon>,
    
    /// S-framework integration
    s_framework: Arc<Mutex<SFramework>>,
    
    /// Catalyst efficiency
    efficiency: f64,
    
    /// System status
    is_active: bool,
}

impl InformationCatalyst {
    pub fn new(id: String, s_framework: &SFramework) -> Self {
        Self {
            id,
            bmds: HashMap::new(),
            s_framework: Arc::new(Mutex::new(s_framework.clone())),
            efficiency: 0.0,
            is_active: false,
        }
    }
    
    /// Add BMD to catalyst system
    pub fn add_bmd(&mut self, mut bmd: BiologicalMaxwellDemon) -> Result<(), BMDError> {
        bmd.enable_s_framework_integration();
        let bmd_id = bmd.id.clone();
        self.bmds.insert(bmd_id, bmd);
        Ok(())
    }
    
    /// Activate information catalyst system
    pub fn activate_catalyst(&mut self) -> Result<(), BMDError> {
        self.is_active = true;
        
        // Activate all BMDs
        for bmd in self.bmds.values_mut() {
            bmd.activate()?;
        }
        
        self.calculate_system_efficiency();
        Ok(())
    }
    
    /// Process pattern through optimal BMD
    pub fn catalyze_information(&mut self, pattern: InformationPattern) -> Result<InformationPattern, BMDError> {
        if !self.is_active {
            return Err(BMDError::CatalysisFailure(
                "Information catalyst not active".to_string()
            ));
        }
        
        // Find optimal BMD based on pattern category
        let optimal_bmd_id = self.find_optimal_bmd(&pattern)?;
        
        // Process pattern through optimal BMD
        let processed_pattern = self.bmds.get_mut(&optimal_bmd_id)
            .unwrap()
            .process_pattern(pattern)?;
        
        self.calculate_system_efficiency();
        
        Ok(processed_pattern)
    }
    
    /// Find optimal BMD for pattern processing
    fn find_optimal_bmd(&self, pattern: &InformationPattern) -> Result<String, BMDError> {
        if self.bmds.is_empty() {
            return Err(BMDError::CatalysisFailure(
                "No BMDs available for catalysis".to_string()
            ));
        }
        
        // Find BMD with highest efficiency and active status
        let optimal_bmd = self.bmds.iter()
            .filter(|(_, bmd)| bmd.is_active)
            .max_by(|(_, a), (_, b)| a.efficiency.partial_cmp(&b.efficiency).unwrap())
            .map(|(id, _)| id.clone())
            .ok_or_else(|| BMDError::CatalysisFailure(
                "No active BMDs available".to_string()
            ))?;
        
        Ok(optimal_bmd)
    }
    
    /// Calculate system-wide efficiency
    fn calculate_system_efficiency(&mut self) {
        if self.bmds.is_empty() {
            self.efficiency = 0.0;
            return;
        }
        
        let total_efficiency: f64 = self.bmds.values()
            .map(|bmd| bmd.efficiency)
            .sum();
        
        self.efficiency = total_efficiency / self.bmds.len() as f64;
    }
    
    /// Get catalyst system status
    pub fn catalyst_status(&self) -> CatalystStatus {
        let bmd_statuses: Vec<BMDStatus> = self.bmds.values()
            .map(|bmd| bmd.status())
            .collect();
        
        CatalystStatus {
            id: self.id.clone(),
            is_active: self.is_active,
            efficiency: self.efficiency,
            bmd_count: self.bmds.len(),
            bmd_statuses,
        }
    }
}

/// Catalyst system status
#[derive(Debug, Clone)]
pub struct CatalystStatus {
    pub id: String,
    pub is_active: bool,
    pub efficiency: f64,
    pub bmd_count: usize,
    pub bmd_statuses: Vec<BMDStatus>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_information_pattern() {
        let data = b"test pattern data".to_vec();
        let mut pattern = InformationPattern::new(
            "test".to_string(),
            data,
            PatternCategory::CognitiveFrame
        );
        
        pattern.assess_consciousness_significance();
        assert!(pattern.consciousness_significance > 0.0);
    }
    
    #[test]
    fn test_entropy_reducer() {
        let mut reducer = EntropyReducer::new("test".to_string());
        let pattern = InformationPattern::new(
            "test".to_string(),
            b"test".to_vec(),
            PatternCategory::EntropyReduction
        );
        
        let reduction = reducer.reduce_entropy(&pattern).unwrap();
        assert!(reduction >= 0.0);
        assert!(reducer.current_entropy() < 1.0);
    }
    
    #[test]
    fn test_pattern_recognizer() {
        let mut recognizer = PatternRecognizer::new("test".to_string());
        let pattern = InformationPattern::new(
            "test".to_string(),
            b"consciousness frame data".to_vec(),
            PatternCategory::CognitiveFrame
        );
        
        let recognized = recognizer.recognize_pattern(pattern).unwrap();
        assert!(recognized.confidence > 0.0);
    }
    
    #[test]
    fn test_biological_maxwell_demon() {
        let mut bmd = BiologicalMaxwellDemon::new("test_bmd".to_string());
        bmd.activate().unwrap();
        
        let pattern = InformationPattern::new(
            "test".to_string(),
            b"test pattern".to_vec(),
            PatternCategory::CognitiveFrame
        );
        
        let processed = bmd.process_pattern(pattern).unwrap();
        assert!(processed.confidence > 0.0);
        assert!(bmd.efficiency > 0.0);
    }
} 