//! # Awareness System
//!
//! This module implements the awareness system that provides the consciousness substrate
//! with comprehensive sensing, monitoring, and situational awareness capabilities.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use uuid::Uuid;

use super::{ConsciousnessConfig, ConsciousnessError, ConsciousnessResult};

/// Sensory input from various system components
#[derive(Debug, Clone)]
pub struct SensoryInput {
    /// Input ID
    pub id: Uuid,
    
    /// Input type (visual, auditory, tactile, etc.)
    pub input_type: String,
    
    /// Raw sensor data
    pub data: Vec<u8>,
    
    /// Input intensity
    pub intensity: f64,
    
    /// Source component
    pub source: String,
    
    /// Timestamp
    pub timestamp: Instant,
    
    /// Confidence level
    pub confidence: f64,
}

/// Awareness level classification
#[derive(Debug, Clone, PartialEq)]
pub enum AwarenessLevel {
    /// Minimal awareness
    Minimal,
    
    /// Basic awareness
    Basic,
    
    /// Enhanced awareness
    Enhanced,
    
    /// Full awareness
    Full,
    
    /// Hyper awareness
    Hyper,
}

/// Awareness state for different domains
#[derive(Debug, Clone)]
pub struct AwarenessState {
    /// Overall awareness level
    pub overall_level: f64,
    
    /// Domain-specific awareness levels
    pub domain_levels: HashMap<String, f64>,
    
    /// Active sensory channels
    pub active_channels: Vec<String>,
    
    /// Attention focus
    pub attention_focus: Vec<String>,
    
    /// Current processing load
    pub processing_load: f64,
    
    /// Last update timestamp
    pub last_update: Instant,
}

/// Attention mechanism for selective awareness
#[derive(Debug, Clone)]
pub struct AttentionMechanism {
    /// Focus targets
    pub focus_targets: Vec<String>,
    
    /// Attention weights
    pub attention_weights: HashMap<String, f64>,
    
    /// Attention span
    pub attention_span: Duration,
    
    /// Distraction threshold
    pub distraction_threshold: f64,
    
    /// Current focus strength
    pub focus_strength: f64,
}

/// Awareness pattern for recognizing important events
#[derive(Debug, Clone)]
pub struct AwarenessPattern {
    /// Pattern ID
    pub id: Uuid,
    
    /// Pattern name
    pub name: String,
    
    /// Pattern signature
    pub signature: Vec<f64>,
    
    /// Confidence threshold
    pub confidence_threshold: f64,
    
    /// Response action
    pub response_action: String,
    
    /// Pattern priority
    pub priority: f64,
}

/// System awareness manager
pub struct AwarenessSystem {
    /// Configuration
    config: ConsciousnessConfig,
    
    /// Current awareness state
    awareness_state: Arc<RwLock<AwarenessState>>,
    
    /// Sensory input buffer
    sensory_buffer: Arc<RwLock<VecDeque<SensoryInput>>>,
    
    /// Attention mechanism
    attention: Arc<RwLock<AttentionMechanism>>,
    
    /// Awareness patterns
    patterns: Arc<RwLock<HashMap<Uuid, AwarenessPattern>>>,
    
    /// Sensory processing threads
    sensory_processors: Arc<RwLock<HashMap<String, SensoryProcessor>>>,
    
    /// Pattern matching engine
    pattern_matcher: Arc<PatternMatcher>,
    
    /// Awareness history
    awareness_history: Arc<RwLock<VecDeque<AwarenessState>>>,
    
    /// Initialization status
    initialized: Arc<Mutex<bool>>,
}

impl AwarenessSystem {
    /// Create new awareness system
    pub fn new(config: &ConsciousnessConfig) -> ConsciousnessResult<Self> {
        let initial_state = AwarenessState {
            overall_level: 1.0,
            domain_levels: HashMap::new(),
            active_channels: Vec::new(),
            attention_focus: Vec::new(),
            processing_load: 0.0,
            last_update: Instant::now(),
        };
        
        let initial_attention = AttentionMechanism {
            focus_targets: Vec::new(),
            attention_weights: HashMap::new(),
            attention_span: Duration::from_secs(10),
            distraction_threshold: 0.7,
            focus_strength: 1.0,
        };
        
        Ok(Self {
            config: config.clone(),
            awareness_state: Arc::new(RwLock::new(initial_state)),
            sensory_buffer: Arc::new(RwLock::new(VecDeque::new())),
            attention: Arc::new(RwLock::new(initial_attention)),
            patterns: Arc::new(RwLock::new(HashMap::new())),
            sensory_processors: Arc::new(RwLock::new(HashMap::new())),
            pattern_matcher: Arc::new(PatternMatcher::new()),
            awareness_history: Arc::new(RwLock::new(VecDeque::new())),
            initialized: Arc::new(Mutex::new(false)),
        })
    }
    
    /// Initialize awareness system
    pub async fn initialize(&self) -> ConsciousnessResult<()> {
        let mut initialized = self.initialized.lock().await;
        if *initialized {
            return Ok(());
        }
        
        tracing::info!("Initializing awareness system");
        
        // Initialize sensory processors
        self.initialize_sensory_processors().await?;
        
        // Initialize awareness patterns
        self.initialize_awareness_patterns().await?;
        
        // Initialize attention mechanism
        self.initialize_attention_mechanism().await?;
        
        // Start background processing
        self.start_background_processing().await?;
        
        *initialized = true;
        tracing::info!("Awareness system initialized successfully");
        Ok(())
    }
    
    /// Shutdown awareness system
    pub async fn shutdown(&self) -> ConsciousnessResult<()> {
        tracing::info!("Shutting down awareness system");
        
        let mut initialized = self.initialized.lock().await;
        *initialized = false;
        
        tracing::info!("Awareness system shutdown complete");
        Ok(())
    }
    
    /// Get current awareness level
    pub async fn get_awareness_level(&self) -> ConsciousnessResult<f64> {
        let state = self.awareness_state.read().unwrap();
        Ok(state.overall_level)
    }
    
    /// Process sensory input
    pub async fn process_input(&self, input: &str) -> ConsciousnessResult<String> {
        let sensory_input = SensoryInput {
            id: Uuid::new_v4(),
            input_type: "textual".to_string(),
            data: input.as_bytes().to_vec(),
            intensity: self.calculate_input_intensity(input).await?,
            source: "consciousness_substrate".to_string(),
            timestamp: Instant::now(),
            confidence: 0.95,
        };
        
        // Add to sensory buffer
        {
            let mut buffer = self.sensory_buffer.write().unwrap();
            buffer.push_back(sensory_input.clone());
            
            // Limit buffer size
            if buffer.len() > 10000 {
                buffer.pop_front();
            }
        }
        
        // Process through attention mechanism
        let processed_input = self.apply_attention_filter(&sensory_input).await?;
        
        // Pattern matching
        let pattern_matches = self.pattern_matcher.match_patterns(&processed_input, &self.patterns).await?;
        
        // Generate awareness response
        let response = self.generate_awareness_response(&processed_input, &pattern_matches).await?;
        
        // Update awareness state
        self.update_awareness_state(&sensory_input, &pattern_matches).await?;
        
        Ok(response)
    }
    
    /// Update awareness
    pub async fn update_awareness(&self) -> ConsciousnessResult<()> {
        // Process pending sensory inputs
        self.process_sensory_buffer().await?;
        
        // Update attention mechanism
        self.update_attention_mechanism().await?;
        
        // Update domain-specific awareness
        self.update_domain_awareness().await?;
        
        // Calculate overall awareness level
        self.calculate_overall_awareness().await?;
        
        // Store awareness history
        self.store_awareness_history().await?;
        
        Ok(())
    }
    
    /// Focus attention on specific targets
    pub async fn focus_attention(&self, targets: Vec<String>) -> ConsciousnessResult<()> {
        let mut attention = self.attention.write().unwrap();
        
        attention.focus_targets = targets.clone();
        attention.focus_strength = 1.0;
        
        // Update attention weights
        for target in targets {
            attention.attention_weights.insert(target, 1.0);
        }
        
        tracing::debug!("Focused attention on {} targets", attention.focus_targets.len());
        Ok(())
    }
    
    /// Add awareness pattern
    pub async fn add_pattern(&self, pattern: AwarenessPattern) -> ConsciousnessResult<()> {
        {
            let mut patterns = self.patterns.write().unwrap();
            patterns.insert(pattern.id, pattern.clone());
        }
        
        tracing::debug!("Added awareness pattern: {}", pattern.name);
        Ok(())
    }
    
    /// Get awareness state
    pub async fn get_awareness_state(&self) -> ConsciousnessResult<AwarenessState> {
        let state = self.awareness_state.read().unwrap();
        Ok(state.clone())
    }
    
    /// Calculate input intensity
    async fn calculate_input_intensity(&self, input: &str) -> ConsciousnessResult<f64> {
        // Simple intensity calculation based on input characteristics
        let length_factor = (input.len() as f64).ln() / 10.0;
        let complexity_factor = self.calculate_text_complexity(input);
        let urgency_factor = self.detect_urgency_indicators(input);
        
        let intensity = (length_factor + complexity_factor + urgency_factor) / 3.0;
        Ok(intensity.min(1.0).max(0.0))
    }
    
    /// Calculate text complexity
    fn calculate_text_complexity(&self, text: &str) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        let avg_word_length = if words.is_empty() {
            0.0
        } else {
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / words.len() as f64
        };
        
        let sentence_count = text.matches('.').count() + text.matches('!').count() + text.matches('?').count();
        let sentence_complexity = if sentence_count > 0 {
            words.len() as f64 / sentence_count as f64
        } else {
            words.len() as f64
        };
        
        ((avg_word_length + sentence_complexity) / 20.0).min(1.0)
    }
    
    /// Detect urgency indicators
    fn detect_urgency_indicators(&self, text: &str) -> f64 {
        let urgency_words = ["urgent", "emergency", "critical", "immediate", "alert", "warning"];
        let exclamation_count = text.matches('!').count();
        let caps_count = text.chars().filter(|c| c.is_uppercase()).count();
        
        let urgency_word_count = urgency_words.iter()
            .map(|&word| text.to_lowercase().matches(word).count())
            .sum::<usize>();
        
        let urgency_score = (urgency_word_count as f64 * 0.3 + 
                           exclamation_count as f64 * 0.1 + 
                           caps_count as f64 * 0.01).min(1.0);
        
        urgency_score
    }
    
    /// Apply attention filter to sensory input
    async fn apply_attention_filter(&self, input: &SensoryInput) -> ConsciousnessResult<SensoryInput> {
        let attention = self.attention.read().unwrap();
        
        let mut filtered_input = input.clone();
        
        // Calculate attention weight for this input
        let attention_weight = self.calculate_attention_weight(input, &attention).await?;
        
        // Modify input intensity based on attention
        filtered_input.intensity *= attention_weight;
        
        Ok(filtered_input)
    }
    
    /// Calculate attention weight for input
    async fn calculate_attention_weight(&self, input: &SensoryInput, attention: &AttentionMechanism) -> ConsciousnessResult<f64> {
        let mut weight = 1.0;
        
        // Check if input matches focus targets
        for target in &attention.focus_targets {
            if input.source.contains(target) || input.input_type.contains(target) {
                weight *= attention.focus_strength;
                break;
            }
        }
        
        // Apply attention weights
        for (pattern, pattern_weight) in &attention.attention_weights {
            if String::from_utf8_lossy(&input.data).contains(pattern) {
                weight *= pattern_weight;
            }
        }
        
        Ok(weight.min(2.0).max(0.1)) // Limit weight range
    }
    
    /// Generate awareness response
    async fn generate_awareness_response(
        &self,
        input: &SensoryInput,
        pattern_matches: &[PatternMatch],
    ) -> ConsciousnessResult<String> {
        let input_text = String::from_utf8_lossy(&input.data);
        
        if pattern_matches.is_empty() {
            // No patterns matched - basic awareness response
            Ok(format!("Aware of: {}", input_text))
        } else {
            // Patterns matched - enhanced awareness response
            let best_match = pattern_matches.iter()
                .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                .unwrap();
            
            Ok(format!("Recognized pattern '{}' with confidence {:.3}: {}", 
                      best_match.pattern_name, best_match.confidence, input_text))
        }
    }
    
    /// Update awareness state
    async fn update_awareness_state(
        &self,
        input: &SensoryInput,
        pattern_matches: &[PatternMatch],
    ) -> ConsciousnessResult<()> {
        let mut state = self.awareness_state.write().unwrap();
        
        // Update processing load
        state.processing_load = (state.processing_load + input.intensity * 0.1).min(1.0);
        
        // Update domain levels
        let domain = input.input_type.clone();
        let current_level = state.domain_levels.get(&domain).unwrap_or(&0.5);
        let new_level = (current_level + input.intensity * 0.1).min(1.0);
        state.domain_levels.insert(domain.clone(), new_level);
        
        // Update active channels
        if !state.active_channels.contains(&input.source) {
            state.active_channels.push(input.source.clone());
        }
        
        // Update attention focus based on patterns
        for pattern_match in pattern_matches {
            if pattern_match.confidence > 0.8 && !state.attention_focus.contains(&pattern_match.pattern_name) {
                state.attention_focus.push(pattern_match.pattern_name.clone());
            }
        }
        
        state.last_update = Instant::now();
        
        Ok(())
    }
    
    /// Initialize sensory processors
    async fn initialize_sensory_processors(&self) -> ConsciousnessResult<()> {
        let mut processors = self.sensory_processors.write().unwrap();
        
        processors.insert("textual".to_string(), SensoryProcessor::new("textual"));
        processors.insert("numerical".to_string(), SensoryProcessor::new("numerical"));
        processors.insert("temporal".to_string(), SensoryProcessor::new("temporal"));
        processors.insert("spatial".to_string(), SensoryProcessor::new("spatial"));
        
        Ok(())
    }
    
    /// Initialize awareness patterns
    async fn initialize_awareness_patterns(&self) -> ConsciousnessResult<()> {
        let patterns = vec![
            AwarenessPattern {
                id: Uuid::new_v4(),
                name: "high_priority".to_string(),
                signature: vec![0.8, 0.9, 0.7],
                confidence_threshold: 0.8,
                response_action: "focus_attention".to_string(),
                priority: 0.9,
            },
            AwarenessPattern {
                id: Uuid::new_v4(),
                name: "emergency".to_string(),
                signature: vec![1.0, 0.8, 0.9],
                confidence_threshold: 0.9,
                response_action: "immediate_attention".to_string(),
                priority: 1.0,
            },
            AwarenessPattern {
                id: Uuid::new_v4(),
                name: "learning_opportunity".to_string(),
                signature: vec![0.6, 0.7, 0.8],
                confidence_threshold: 0.7,
                response_action: "enhance_learning".to_string(),
                priority: 0.6,
            },
        ];
        
        for pattern in patterns {
            self.add_pattern(pattern).await?;
        }
        
        Ok(())
    }
    
    /// Initialize attention mechanism
    async fn initialize_attention_mechanism(&self) -> ConsciousnessResult<()> {
        let mut attention = self.attention.write().unwrap();
        
        attention.focus_targets = vec![
            "consciousness".to_string(),
            "learning".to_string(),
            "coherence".to_string(),
        ];
        
        attention.attention_weights.insert("consciousness".to_string(), 1.0);
        attention.attention_weights.insert("learning".to_string(), 0.8);
        attention.attention_weights.insert("coherence".to_string(), 0.9);
        
        Ok(())
    }
    
    /// Start background processing
    async fn start_background_processing(&self) -> ConsciousnessResult<()> {
        // In a real implementation, this would start background tasks
        tracing::debug!("Started awareness background processing");
        Ok(())
    }
    
    /// Process sensory buffer
    async fn process_sensory_buffer(&self) -> ConsciousnessResult<()> {
        let mut buffer = self.sensory_buffer.write().unwrap();
        let inputs_to_process: Vec<SensoryInput> = buffer.drain(..).collect();
        drop(buffer);
        
        for input in inputs_to_process {
            // Process each input through the awareness pipeline
            let _response = self.process_input(&String::from_utf8_lossy(&input.data)).await?;
        }
        
        Ok(())
    }
    
    /// Update attention mechanism
    async fn update_attention_mechanism(&self) -> ConsciousnessResult<()> {
        let mut attention = self.attention.write().unwrap();
        
        // Decay attention weights over time
        for weight in attention.attention_weights.values_mut() {
            *weight *= 0.99; // Gradual decay
        }
        
        // Update focus strength based on processing load
        let state = self.awareness_state.read().unwrap();
        attention.focus_strength = (2.0 - state.processing_load).max(0.5).min(2.0);
        
        Ok(())
    }
    
    /// Update domain-specific awareness
    async fn update_domain_awareness(&self) -> ConsciousnessResult<()> {
        let mut state = self.awareness_state.write().unwrap();
        
        // Decay domain levels over time
        for level in state.domain_levels.values_mut() {
            *level *= 0.995; // Gradual decay
        }
        
        Ok(())
    }
    
    /// Calculate overall awareness level
    async fn calculate_overall_awareness(&self) -> ConsciousnessResult<()> {
        let mut state = self.awareness_state.write().unwrap();
        
        if state.domain_levels.is_empty() {
            state.overall_level = 0.5;
        } else {
            let sum: f64 = state.domain_levels.values().sum();
            let avg = sum / state.domain_levels.len() as f64;
            state.overall_level = avg;
        }
        
        Ok(())
    }
    
    /// Store awareness history
    async fn store_awareness_history(&self) -> ConsciousnessResult<()> {
        let state = self.awareness_state.read().unwrap();
        let mut history = self.awareness_history.write().unwrap();
        
        history.push_back(state.clone());
        
        // Limit history size
        if history.len() > 1000 {
            history.pop_front();
        }
        
        Ok(())
    }
}

/// Sensory processor for different input types
#[derive(Debug, Clone)]
pub struct SensoryProcessor {
    pub processor_type: String,
    pub processing_capacity: f64,
    pub current_load: f64,
}

impl SensoryProcessor {
    pub fn new(processor_type: &str) -> Self {
        Self {
            processor_type: processor_type.to_string(),
            processing_capacity: 1.0,
            current_load: 0.0,
        }
    }
}

/// Pattern matcher for awareness patterns
pub struct PatternMatcher {
    matching_threshold: f64,
}

impl PatternMatcher {
    pub fn new() -> Self {
        Self {
            matching_threshold: 0.7,
        }
    }
    
    pub async fn match_patterns(
        &self,
        input: &SensoryInput,
        patterns: &Arc<RwLock<HashMap<Uuid, AwarenessPattern>>>,
    ) -> ConsciousnessResult<Vec<PatternMatch>> {
        let patterns_guard = patterns.read().unwrap();
        let mut matches = Vec::new();
        
        let input_signature = self.extract_signature(input);
        
        for pattern in patterns_guard.values() {
            let confidence = self.calculate_pattern_confidence(&input_signature, &pattern.signature);
            
            if confidence >= pattern.confidence_threshold {
                matches.push(PatternMatch {
                    pattern_id: pattern.id,
                    pattern_name: pattern.name.clone(),
                    confidence,
                    response_action: pattern.response_action.clone(),
                });
            }
        }
        
        // Sort by confidence
        matches.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        Ok(matches)
    }
    
    fn extract_signature(&self, input: &SensoryInput) -> Vec<f64> {
        // Simple signature extraction based on input characteristics
        vec![
            input.intensity,
            input.confidence,
            (input.data.len() as f64 / 1000.0).min(1.0),
        ]
    }
    
    fn calculate_pattern_confidence(&self, input_signature: &[f64], pattern_signature: &[f64]) -> f64 {
        if input_signature.len() != pattern_signature.len() {
            return 0.0;
        }
        
        let mut distance = 0.0;
        for (a, b) in input_signature.iter().zip(pattern_signature.iter()) {
            distance += (a - b).powi(2);
        }
        
        let normalized_distance = distance.sqrt() / (input_signature.len() as f64).sqrt();
        (1.0 - normalized_distance).max(0.0)
    }
}

/// Pattern match result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern_id: Uuid,
    pub pattern_name: String,
    pub confidence: f64,
    pub response_action: String,
} 