"""
Network Understanding System

This module implements the revolutionary network-of-information-about-information
architecture where each new piece of data influences how ALL future information
is stored, validating that understanding accumulates and evolves.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
import json
import time
import networkx as nx
from enum import Enum
import pickle


class UnderstandingLevel(Enum):
    """Levels of understanding in the network."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class InformationNode:
    """Represents a node in the understanding network."""
    data: str
    understanding_level: UnderstandingLevel
    connections: Set[str] = field(default_factory=set)
    influence_weight: float = 1.0
    storage_pattern: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnderstandingEvolution:
    """Tracks how understanding evolves over time."""
    timestamp: float
    information_count: int
    understanding_metrics: Dict[str, float]
    storage_efficiency: float
    network_complexity: float
    adaptation_events: List[str]


@dataclass
class NetworkState:
    """Represents the current state of the understanding network."""
    nodes: Dict[str, InformationNode]
    connections: Dict[str, Set[str]]
    understanding_history: List[UnderstandingEvolution]
    current_understanding_level: UnderstandingLevel
    storage_patterns: Dict[str, Dict[str, Any]]


class UnderstandingNetwork:
    """
    Advanced Network Understanding System
    
    This system validates the core principle that each new piece of information
    influences how ALL future information is stored, creating an evolving
    network of understanding about understanding itself.
    
    Key Capabilities:
    1. Evolving storage patterns based on accumulated understanding
    2. Information-about-information network construction
    3. Adaptive compression and retrieval optimization
    4. Self-improving understanding metrics
    """
    
    def __init__(self, adaptation_threshold: float = 0.1):
        """
        Initialize the understanding network system.
        
        Args:
            adaptation_threshold: Threshold for triggering network adaptations
        """
        self.adaptation_threshold = adaptation_threshold
        self.network_graph: nx.DiGraph = nx.DiGraph()
        self.information_nodes: Dict[str, InformationNode] = {}
        self.understanding_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.evolution_history: List[UnderstandingEvolution] = []
        self.current_storage_strategy: Dict[str, Any] = {}
        self.influence_propagation_cache: Dict[str, Dict[str, float]] = {}
        
    def ingest_information(self, data: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Ingest new information and adapt the entire network based on understanding.
        
        This validates that storage and understanding co-evolve by demonstrating
        how each new piece of information changes the storage strategy for
        ALL information in the system.
        
        Args:
            data: New information to ingest
            context: Optional context for the information
            
        Returns:
            Analysis of how the network adapted to the new information
        """
        
        if context is None:
            context = {}
            
        print(f"Ingesting new information: '{data[:50]}...'")
        
        # Step 1: Analyze new information in context of existing network
        pre_ingestion_state = self._capture_network_state()
        
        # Step 2: Determine understanding level of new information
        understanding_level = self._assess_understanding_level(data)
        print(f"Assessed understanding level: {understanding_level.value}")
        
        # Step 3: Create information node and integrate into network
        node_id = self._create_information_node(data, understanding_level, context)
        print(f"Created node: {node_id}")
        
        # Step 4: Propagate influence throughout network
        influence_changes = self._propagate_understanding_influence(node_id)
        print(f"Propagated influence to {len(influence_changes)} nodes")
        
        # Step 5: Adapt storage patterns based on new understanding
        storage_adaptations = self._adapt_storage_patterns()
        print(f"Adapted {len(storage_adaptations)} storage patterns")
        
        # Step 6: Update network relationships
        new_connections = self._update_network_relationships(node_id)
        print(f"Created {len(new_connections)} new connections")
        
        # Step 7: Evolve understanding metrics
        understanding_evolution = self._evolve_understanding_metrics()
        
        # Step 8: Record evolution event
        post_ingestion_state = self._capture_network_state()
        evolution_event = self._create_evolution_event(
            pre_ingestion_state, 
            post_ingestion_state,
            node_id
        )
        self.evolution_history.append(evolution_event)
        
        return {
            "node_id": node_id,
            "understanding_level": understanding_level.value,
            "influence_changes": influence_changes,
            "storage_adaptations": storage_adaptations,
            "new_connections": new_connections,
            "understanding_evolution": understanding_evolution,
            "network_impact": self._calculate_network_impact(pre_ingestion_state, post_ingestion_state)
        }
    
    def demonstrate_understanding_accumulation(self, information_sequence: List[str]) -> Dict[str, Any]:
        """
        Demonstrate how understanding accumulates and influences future storage.
        
        This validates that the system exhibits genuine learning by showing
        measurable improvements in storage efficiency and understanding metrics.
        
        Args:
            information_sequence: Sequence of information to ingest
            
        Returns:
            Comprehensive analysis of understanding accumulation
        """
        
        print(f"Demonstrating understanding accumulation with {len(information_sequence)} information pieces")
        
        accumulation_results = []
        efficiency_progression = []
        understanding_progression = []
        
        for i, info in enumerate(information_sequence):
            print(f"\nProcessing item {i+1}/{len(information_sequence)}")
            
            # Capture state before ingestion
            pre_state = self._capture_network_state()
            
            # Ingest information
            ingestion_result = self.ingest_information(info)
            
            # Capture state after ingestion
            post_state = self._capture_network_state()
            
            # Calculate progression metrics
            efficiency = self._calculate_storage_efficiency()
            understanding = self._calculate_overall_understanding()
            
            efficiency_progression.append(efficiency)
            understanding_progression.append(understanding)
            
            accumulation_results.append({
                "sequence_index": i,
                "information": info[:50] + "..." if len(info) > 50 else info,
                "storage_efficiency": efficiency,
                "understanding_score": understanding,
                "network_size": len(self.information_nodes),
                "adaptation_events": len(ingestion_result["storage_adaptations"]),
                "influence_propagated": len(ingestion_result["influence_changes"])
            })
        
        # Analyze learning progression
        learning_analysis = self._analyze_learning_progression(
            efficiency_progression, 
            understanding_progression
        )
        
        return {
            "accumulation_results": accumulation_results,
            "learning_progression": learning_analysis,
            "final_network_state": self._get_network_summary(),
            "understanding_evolution": self.evolution_history,
            "validation_metrics": self._validate_understanding_accumulation()
        }
    
    def _assess_understanding_level(self, data: str) -> UnderstandingLevel:
        """
        Assess the understanding level required for new information.
        
        This determines how the new information should influence the network.
        """
        
        # Analyze complexity factors
        length_complexity = min(1.0, len(data) / 1000.0)
        vocabulary_complexity = len(set(data.split())) / len(data.split()) if data.split() else 0
        
        # Check for patterns similar to existing network knowledge
        pattern_familiarity = self._calculate_pattern_familiarity(data)
        
        # Assess conceptual depth
        conceptual_depth = self._assess_conceptual_depth(data)
        
        # Combine factors to determine understanding level
        complexity_score = (
            0.3 * length_complexity +
            0.2 * vocabulary_complexity +
            0.3 * (1.0 - pattern_familiarity) +  # Novelty increases complexity
            0.2 * conceptual_depth
        )
        
        if complexity_score < 0.3:
            return UnderstandingLevel.BASIC
        elif complexity_score < 0.6:
            return UnderstandingLevel.INTERMEDIATE
        elif complexity_score < 0.8:
            return UnderstandingLevel.ADVANCED
        else:
            return UnderstandingLevel.EXPERT
    
    def _calculate_pattern_familiarity(self, data: str) -> float:
        """
        Calculate how similar new data is to existing network patterns.
        
        This demonstrates how accumulated understanding influences new information processing.
        """
        
        if not self.understanding_patterns:
            return 0.0
        
        data_tokens = set(data.lower().split())
        max_similarity = 0.0
        
        for pattern_type, patterns in self.understanding_patterns.items():
            for pattern_key, pattern_data in patterns.items():
                if "tokens" in pattern_data:
                    pattern_tokens = set(pattern_data["tokens"])
                    
                    if data_tokens and pattern_tokens:
                        intersection = len(data_tokens & pattern_tokens)
                        union = len(data_tokens | pattern_tokens)
                        similarity = intersection / union if union > 0 else 0.0
                        
                        max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _assess_conceptual_depth(self, data: str) -> float:
        """
        Assess the conceptual depth of new information.
        
        This determines how much the new information should influence
        the understanding network.
        """
        
        data_lower = data.lower()
        
        # Check for abstract concepts
        abstract_indicators = ["theory", "concept", "principle", "framework", "model", "paradigm"]
        abstract_score = sum(1 for indicator in abstract_indicators if indicator in data_lower) / len(abstract_indicators)
        
        # Check for relational complexity
        relational_indicators = ["relationship", "connection", "interaction", "influence", "causation"]
        relational_score = sum(1 for indicator in relational_indicators if indicator in data_lower) / len(relational_indicators)
        
        # Check for meta-cognitive elements
        meta_indicators = ["understand", "comprehend", "analyze", "synthesize", "evaluate"]
        meta_score = sum(1 for indicator in meta_indicators if indicator in data_lower) / len(meta_indicators)
        
        return (abstract_score + relational_score + meta_score) / 3.0
    
    def _create_information_node(self, 
                                data: str, 
                                understanding_level: UnderstandingLevel,
                                context: Dict[str, Any]) -> str:
        """
        Create new information node and integrate into network.
        
        The storage pattern is influenced by the current state of understanding.
        """
        
        # Generate unique node ID
        node_id = f"node_{len(self.information_nodes)}_{int(time.time() * 1000) % 10000}"
        
        # Determine storage pattern based on current network understanding
        storage_pattern = self._determine_storage_pattern(data, understanding_level)
        
        # Calculate influence weight based on understanding level and network state
        influence_weight = self._calculate_influence_weight(understanding_level)
        
        # Create information node
        node = InformationNode(
            data=data,
            understanding_level=understanding_level,
            connections=set(),
            influence_weight=influence_weight,
            storage_pattern=storage_pattern,
            creation_time=time.time(),
            metadata={
                "context": context,
                "network_size_at_creation": len(self.information_nodes),
                "understanding_level_at_creation": self._calculate_overall_understanding()
            }
        )
        
        # Add to network
        self.information_nodes[node_id] = node
        self.network_graph.add_node(node_id, **node.__dict__)
        
        return node_id
    
    def _determine_storage_pattern(self, data: str, understanding_level: UnderstandingLevel) -> Dict[str, Any]:
        """
        Determine storage pattern based on accumulated understanding.
        
        This validates that storage decisions are influenced by network knowledge.
        """
        
        # Base storage pattern
        pattern = {
            "compression_strategy": "basic",
            "indexing_method": "linear",
            "retrieval_optimization": "none"
        }
        
        # Adapt pattern based on understanding level
        if understanding_level == UnderstandingLevel.EXPERT:
            pattern["compression_strategy"] = "meta_cascade"
            pattern["indexing_method"] = "semantic_network"
            pattern["retrieval_optimization"] = "direct_navigation"
        elif understanding_level == UnderstandingLevel.ADVANCED:
            pattern["compression_strategy"] = "context_aware"
            pattern["indexing_method"] = "hierarchical"
            pattern["retrieval_optimization"] = "relationship_based"
        elif understanding_level == UnderstandingLevel.INTERMEDIATE:
            pattern["compression_strategy"] = "pattern_based"
            pattern["indexing_method"] = "clustered"
            pattern["retrieval_optimization"] = "context_hinted"
        
        # Further adapt based on network patterns
        if self.current_storage_strategy:
            # Use accumulated understanding to refine pattern
            if "preferred_compression" in self.current_storage_strategy:
                pattern["compression_strategy"] = self.current_storage_strategy["preferred_compression"]
            
            if "optimal_indexing" in self.current_storage_strategy:
                pattern["indexing_method"] = self.current_storage_strategy["optimal_indexing"]
        
        return pattern
    
    def _calculate_influence_weight(self, understanding_level: UnderstandingLevel) -> float:
        """
        Calculate how much influence new information should have on the network.
        """
        
        level_weights = {
            UnderstandingLevel.BASIC: 0.1,
            UnderstandingLevel.INTERMEDIATE: 0.3,
            UnderstandingLevel.ADVANCED: 0.6,
            UnderstandingLevel.EXPERT: 1.0
        }
        
        base_weight = level_weights[understanding_level]
        
        # Adjust based on current network understanding
        network_understanding = self._calculate_overall_understanding()
        
        # Higher understanding network can better evaluate new information influence
        adjusted_weight = base_weight * (0.5 + 0.5 * network_understanding)
        
        return adjusted_weight
    
    def _propagate_understanding_influence(self, node_id: str) -> Dict[str, float]:
        """
        Propagate understanding influence throughout the network.
        
        This validates that new information affects ALL information storage.
        """
        
        if node_id not in self.information_nodes:
            return {}
        
        new_node = self.information_nodes[node_id]
        influence_changes = {}
        
        # Propagate influence to all existing nodes
        for existing_id, existing_node in self.information_nodes.items():
            if existing_id != node_id:
                
                # Calculate influence based on similarity and new node weight
                similarity = self._calculate_node_similarity(new_node, existing_node)
                influence = new_node.influence_weight * similarity
                
                if influence > self.adaptation_threshold:
                    # Apply influence to existing node's storage pattern
                    self._apply_influence_to_node(existing_node, new_node, influence)
                    influence_changes[existing_id] = influence
        
        return influence_changes
    
    def _calculate_node_similarity(self, node_a: InformationNode, node_b: InformationNode) -> float:
        """
        Calculate similarity between information nodes.
        
        This determines how much nodes should influence each other.
        """
        
        # Token-based similarity
        tokens_a = set(node_a.data.lower().split())
        tokens_b = set(node_b.data.lower().split())
        
        if not tokens_a or not tokens_b:
            return 0.0
        
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        token_similarity = intersection / union if union > 0 else 0.0
        
        # Understanding level similarity
        level_similarity = 1.0 - abs(
            list(UnderstandingLevel).index(node_a.understanding_level) -
            list(UnderstandingLevel).index(node_b.understanding_level)
        ) / len(UnderstandingLevel)
        
        # Combined similarity
        return 0.7 * token_similarity + 0.3 * level_similarity
    
    def _apply_influence_to_node(self, 
                                target_node: InformationNode, 
                                source_node: InformationNode, 
                                influence: float) -> None:
        """
        Apply understanding influence from source node to target node.
        
        This demonstrates how accumulated understanding changes existing information storage.
        """
        
        # Adapt storage pattern based on influence
        if influence > 0.5 and source_node.understanding_level.value > target_node.understanding_level.value:
            # Upgrade storage pattern based on higher understanding
            if "compression_strategy" in source_node.storage_pattern:
                target_node.storage_pattern["compression_strategy"] = source_node.storage_pattern["compression_strategy"]
            
            if "retrieval_optimization" in source_node.storage_pattern:
                target_node.storage_pattern["retrieval_optimization"] = source_node.storage_pattern["retrieval_optimization"]
        
        # Update influence weight
        target_node.influence_weight = max(target_node.influence_weight, influence * 0.5)
        
        # Create connection
        target_node.connections.add(f"node_{hash(source_node.data) % 10000}")
        source_node.connections.add(f"node_{hash(target_node.data) % 10000}")
    
    def _adapt_storage_patterns(self) -> Dict[str, Any]:
        """
        Adapt global storage patterns based on network evolution.
        
        This validates that understanding influences system-wide storage decisions.
        """
        
        adaptations = {}
        
        # Analyze current network patterns
        pattern_analysis = self._analyze_network_patterns()
        
        # Update current storage strategy based on analysis
        if pattern_analysis["dominant_compression"] != self.current_storage_strategy.get("preferred_compression"):
            self.current_storage_strategy["preferred_compression"] = pattern_analysis["dominant_compression"]
            adaptations["compression_strategy"] = pattern_analysis["dominant_compression"]
        
        if pattern_analysis["optimal_indexing"] != self.current_storage_strategy.get("optimal_indexing"):
            self.current_storage_strategy["optimal_indexing"] = pattern_analysis["optimal_indexing"]
            adaptations["indexing_method"] = pattern_analysis["optimal_indexing"]
        
        # Update understanding patterns
        self._update_understanding_patterns(pattern_analysis)
        
        return adaptations
    
    def _analyze_network_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns across the entire network.
        
        This identifies emergent patterns that should influence future storage.
        """
        
        compression_strategies = []
        indexing_methods = []
        understanding_levels = []
        
        for node in self.information_nodes.values():
            if "compression_strategy" in node.storage_pattern:
                compression_strategies.append(node.storage_pattern["compression_strategy"])
            
            if "indexing_method" in node.storage_pattern:
                indexing_methods.append(node.storage_pattern["indexing_method"])
            
            understanding_levels.append(node.understanding_level.value)
        
        # Find dominant patterns
        compression_counts = Counter(compression_strategies)
        indexing_counts = Counter(indexing_methods)
        understanding_counts = Counter(understanding_levels)
        
        return {
            "dominant_compression": compression_counts.most_common(1)[0][0] if compression_counts else "basic",
            "optimal_indexing": indexing_counts.most_common(1)[0][0] if indexing_counts else "linear",
            "understanding_distribution": dict(understanding_counts),
            "network_complexity": len(set(compression_strategies + indexing_methods))
        }
    
    def _update_understanding_patterns(self, pattern_analysis: Dict[str, Any]) -> None:
        """
        Update understanding patterns based on network analysis.
        
        This creates the meta-patterns that influence future information processing.
        """
        
        # Update compression patterns
        if "compression" not in self.understanding_patterns:
            self.understanding_patterns["compression"] = {}
        
        compression_pattern = {
            "dominant_strategy": pattern_analysis["dominant_compression"],
            "effectiveness_score": self._calculate_storage_efficiency(),
            "usage_count": sum(1 for node in self.information_nodes.values() 
                             if node.storage_pattern.get("compression_strategy") == pattern_analysis["dominant_compression"]),
            "tokens": set()
        }
        
        # Extract tokens from nodes using this compression strategy
        for node in self.information_nodes.values():
            if node.storage_pattern.get("compression_strategy") == pattern_analysis["dominant_compression"]:
                compression_pattern["tokens"].update(node.data.lower().split())
        
        compression_pattern["tokens"] = list(compression_pattern["tokens"])
        
        self.understanding_patterns["compression"][pattern_analysis["dominant_compression"]] = compression_pattern
    
    def _update_network_relationships(self, node_id: str) -> List[Tuple[str, str]]:
        """
        Update network relationships based on new node integration.
        """
        
        new_connections = []
        
        if node_id not in self.information_nodes:
            return new_connections
        
        new_node = self.information_nodes[node_id]
        
        # Find nodes to connect based on similarity and understanding compatibility
        for existing_id, existing_node in self.information_nodes.items():
            if existing_id != node_id:
                
                similarity = self._calculate_node_similarity(new_node, existing_node)
                
                if similarity > 0.5:  # High similarity threshold for connections
                    # Add graph edge
                    self.network_graph.add_edge(node_id, existing_id, weight=similarity)
                    new_connections.append((node_id, existing_id))
        
        return new_connections
    
    def _evolve_understanding_metrics(self) -> Dict[str, float]:
        """
        Evolve understanding metrics based on network growth.
        """
        
        return {
            "network_understanding": self._calculate_overall_understanding(),
            "storage_efficiency": self._calculate_storage_efficiency(),
            "connection_density": self._calculate_connection_density(),
            "pattern_recognition": self._calculate_pattern_recognition_ability()
        }
    
    def _calculate_overall_understanding(self) -> float:
        """
        Calculate the overall understanding level of the network.
        """
        
        if not self.information_nodes:
            return 0.0
        
        # Weight understanding levels
        level_weights = {
            UnderstandingLevel.BASIC: 0.25,
            UnderstandingLevel.INTERMEDIATE: 0.5,
            UnderstandingLevel.ADVANCED: 0.75,
            UnderstandingLevel.EXPERT: 1.0
        }
        
        total_weighted_understanding = sum(
            level_weights[node.understanding_level] * node.influence_weight
            for node in self.information_nodes.values()
        )
        
        total_weight = sum(node.influence_weight for node in self.information_nodes.values())
        
        return total_weighted_understanding / total_weight if total_weight > 0 else 0.0
    
    def _calculate_storage_efficiency(self) -> float:
        """
        Calculate storage efficiency based on network optimization.
        """
        
        if not self.information_nodes:
            return 0.0
        
        # Count advanced storage patterns
        advanced_patterns = sum(
            1 for node in self.information_nodes.values()
            if node.storage_pattern.get("compression_strategy") in ["meta_cascade", "context_aware"]
        )
        
        efficiency = advanced_patterns / len(self.information_nodes)
        
        # Bonus for network-wide optimization
        if len(self.current_storage_strategy) > 0:
            efficiency *= 1.2
        
        return min(1.0, efficiency)
    
    def _calculate_connection_density(self) -> float:
        """
        Calculate network connection density.
        """
        
        if len(self.information_nodes) < 2:
            return 0.0
        
        actual_connections = self.network_graph.number_of_edges()
        max_possible_connections = len(self.information_nodes) * (len(self.information_nodes) - 1) / 2
        
        return actual_connections / max_possible_connections if max_possible_connections > 0 else 0.0
    
    def _calculate_pattern_recognition_ability(self) -> float:
        """
        Calculate the network's pattern recognition ability.
        """
        
        pattern_count = sum(len(patterns) for patterns in self.understanding_patterns.values())
        node_count = len(self.information_nodes)
        
        if node_count == 0:
            return 0.0
        
        # Pattern recognition improves with more patterns relative to information
        pattern_ratio = pattern_count / node_count
        
        return min(1.0, pattern_ratio)
    
    def _capture_network_state(self) -> NetworkState:
        """
        Capture current network state for comparison.
        """
        
        return NetworkState(
            nodes=dict(self.information_nodes),
            connections={node_id: node.connections.copy() for node_id, node in self.information_nodes.items()},
            understanding_history=self.evolution_history.copy(),
            current_understanding_level=UnderstandingLevel.BASIC,  # Simplified for demo
            storage_patterns=dict(self.current_storage_strategy)
        )
    
    def _create_evolution_event(self, 
                               pre_state: NetworkState, 
                               post_state: NetworkState,
                               trigger_node: str) -> UnderstandingEvolution:
        """
        Create evolution event record.
        """
        
        return UnderstandingEvolution(
            timestamp=time.time(),
            information_count=len(post_state.nodes),
            understanding_metrics=self._evolve_understanding_metrics(),
            storage_efficiency=self._calculate_storage_efficiency(),
            network_complexity=len(self.understanding_patterns),
            adaptation_events=[f"Added node: {trigger_node}", f"Network size: {len(post_state.nodes)}"]
        )
    
    def _calculate_network_impact(self, pre_state: NetworkState, post_state: NetworkState) -> Dict[str, Any]:
        """
        Calculate impact of new information on network.
        """
        
        return {
            "node_count_change": len(post_state.nodes) - len(pre_state.nodes),
            "connection_count_change": sum(len(conns) for conns in post_state.connections.values()) - 
                                    sum(len(conns) for conns in pre_state.connections.values()),
            "storage_pattern_changes": len(post_state.storage_patterns) - len(pre_state.storage_patterns),
            "understanding_improvement": post_state.understanding_history[-1].understanding_metrics["network_understanding"] - 
                                       (pre_state.understanding_history[-1].understanding_metrics["network_understanding"] 
                                        if pre_state.understanding_history else 0.0)
        }
    
    def _analyze_learning_progression(self, 
                                    efficiency_progression: List[float], 
                                    understanding_progression: List[float]) -> Dict[str, Any]:
        """
        Analyze learning progression over information sequence.
        """
        
        if len(efficiency_progression) < 2 or len(understanding_progression) < 2:
            return {"learning_detected": False}
        
        # Calculate trends
        efficiency_trend = np.polyfit(range(len(efficiency_progression)), efficiency_progression, 1)[0]
        understanding_trend = np.polyfit(range(len(understanding_progression)), understanding_progression, 1)[0]
        
        # Detect acceleration in learning
        mid_point = len(efficiency_progression) // 2
        early_efficiency = np.mean(efficiency_progression[:mid_point])
        late_efficiency = np.mean(efficiency_progression[mid_point:])
        
        early_understanding = np.mean(understanding_progression[:mid_point])
        late_understanding = np.mean(understanding_progression[mid_point:])
        
        return {
            "learning_detected": efficiency_trend > 0 and understanding_trend > 0,
            "efficiency_improvement": late_efficiency - early_efficiency,
            "understanding_improvement": late_understanding - early_understanding,
            "efficiency_trend": efficiency_trend,
            "understanding_trend": understanding_trend,
            "acceleration_phase": mid_point,
            "final_efficiency": efficiency_progression[-1],
            "final_understanding": understanding_progression[-1]
        }
    
    def _get_network_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive network summary.
        """
        
        return {
            "total_nodes": len(self.information_nodes),
            "total_connections": self.network_graph.number_of_edges(),
            "understanding_distribution": Counter(node.understanding_level.value for node in self.information_nodes.values()),
            "storage_patterns": Counter(
                node.storage_pattern.get("compression_strategy", "basic") 
                for node in self.information_nodes.values()
            ),
            "network_metrics": self._evolve_understanding_metrics(),
            "evolution_events": len(self.evolution_history)
        }
    
    def _validate_understanding_accumulation(self) -> Dict[str, Any]:
        """
        Validate that understanding actually accumulates and influences storage.
        """
        
        if len(self.evolution_history) < 2:
            return {"validation_possible": False}
        
        # Check for improvement over time
        first_metrics = self.evolution_history[0].understanding_metrics
        last_metrics = self.evolution_history[-1].understanding_metrics
        
        understanding_improved = last_metrics["network_understanding"] > first_metrics["network_understanding"]
        efficiency_improved = last_metrics["storage_efficiency"] > first_metrics["storage_efficiency"]
        
        # Check for pattern development
        pattern_growth = len(self.understanding_patterns) > 0
        
        # Check for storage adaptation
        storage_adaptation = len(self.current_storage_strategy) > 0
        
        return {
            "validation_possible": True,
            "understanding_improved": understanding_improved,
            "efficiency_improved": efficiency_improved,
            "patterns_developed": pattern_growth,
            "storage_adapted": storage_adaptation,
            "overall_validation": all([understanding_improved, efficiency_improved, pattern_growth, storage_adaptation])
        }
    
    def export_network(self) -> str:
        """
        Export network state for analysis.
        """
        
        export_data = {
            "network_summary": self._get_network_summary(),
            "evolution_history": [
                {
                    "timestamp": event.timestamp,
                    "information_count": event.information_count,
                    "understanding_metrics": event.understanding_metrics,
                    "storage_efficiency": event.storage_efficiency
                }
                for event in self.evolution_history
            ],
            "understanding_patterns": {
                pattern_type: {
                    pattern_key: {k: v for k, v in pattern_data.items() if k != "tokens"}
                    for pattern_key, pattern_data in patterns.items()
                }
                for pattern_type, patterns in self.understanding_patterns.items()
            },
            "validation_results": self._validate_understanding_accumulation()
        }
        
        return json.dumps(export_data, indent=2)
    
    def get_node_influence_analysis(self, node_id: str) -> Dict[str, Any]:
        """
        Analyze how a specific node influences the network.
        """
        
        if node_id not in self.information_nodes:
            return {"error": "Node not found"}
        
        node = self.information_nodes[node_id]
        
        # Calculate influence reach
        influenced_nodes = []
        for other_id, other_node in self.information_nodes.items():
            if other_id != node_id:
                similarity = self._calculate_node_similarity(node, other_node)
                if similarity * node.influence_weight > self.adaptation_threshold:
                    influenced_nodes.append((other_id, similarity))
        
        return {
            "node_id": node_id,
            "understanding_level": node.understanding_level.value,
            "influence_weight": node.influence_weight,
            "direct_connections": len(node.connections),
            "influenced_nodes": len(influenced_nodes),
            "influence_reach": influenced_nodes,
            "storage_pattern": node.storage_pattern,
            "creation_time": node.creation_time,
            "network_position": self._calculate_network_centrality(node_id)
        }
    
    def _calculate_network_centrality(self, node_id: str) -> Dict[str, float]:
        """
        Calculate centrality metrics for a node.
        """
        
        if node_id not in self.network_graph:
            return {}
        
        centrality_metrics = {}
        
        try:
            centrality_metrics["betweenness"] = nx.betweenness_centrality(self.network_graph)[node_id]
            centrality_metrics["closeness"] = nx.closeness_centrality(self.network_graph)[node_id]
            centrality_metrics["degree"] = nx.degree_centrality(self.network_graph)[node_id]
        except:
            # Handle disconnected graphs
            centrality_metrics = {"betweenness": 0.0, "closeness": 0.0, "degree": 0.0}
        
        return centrality_metrics
