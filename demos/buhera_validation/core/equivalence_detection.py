"""
Equivalence Class Detection System

This module implements sophisticated detection of context-dependent symbols
and their multiple meanings, validating that understanding is required for
optimal information processing.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
from enum import Enum


class SemanticContext(Enum):
    """Enumeration of semantic context types."""
    MATHEMATICAL = "mathematical"
    PROCEDURAL = "procedural"
    INDEXING = "indexing"
    QUANTITATIVE = "quantitative"
    TEMPORAL = "temporal"
    LOGICAL = "logical"
    RELATIONAL = "relational"
    GENERAL = "general"


@dataclass
class SymbolOccurrence:
    """Records a single occurrence of a symbol with its context."""
    symbol: str
    position: int
    context_before: str
    context_after: str
    semantic_context: SemanticContext
    confidence: float


@dataclass
class ContextPattern:
    """Represents a pattern for identifying specific contexts."""
    name: str
    regex_patterns: List[str]
    keywords: List[str]
    context_type: SemanticContext
    weight: float


@dataclass
class EquivalenceRelation:
    """Defines an equivalence relationship between symbols."""
    symbol_a: str
    symbol_b: str
    context_type: SemanticContext
    equivalence_strength: float
    examples: List[Tuple[str, str]]


class EquivalenceDetector:
    """
    Advanced Equivalence Class Detection System
    
    This system demonstrates that understanding context-dependent meanings
    is computationally necessary for optimal information processing.
    
    Key Capabilities:
    1. Semantic context classification
    2. Multi-meaning symbol detection
    3. Equivalence relationship discovery
    4. Context-dependent storage optimization
    """
    
    def __init__(self, min_confidence: float = 0.6):
        """
        Initialize the equivalence detection system.
        
        Args:
            min_confidence: Minimum confidence threshold for equivalence detection
        """
        self.min_confidence = min_confidence
        self.context_patterns = self._initialize_context_patterns()
        self.symbol_occurrences: List[SymbolOccurrence] = []
        self.equivalence_relations: List[EquivalenceRelation] = []
        self.semantic_network: Dict[str, Set[str]] = defaultdict(set)
        
    def analyze_data(self, data: str) -> Dict[str, Any]:
        """
        Analyze data to detect equivalence classes and context-dependent meanings.
        
        This method validates that understanding is computationally required
        by demonstrating superior compression through semantic analysis.
        
        Args:
            data: Input text data to analyze
            
        Returns:
            Comprehensive analysis results with equivalence classes and metrics
        """
        
        print("Starting equivalence detection analysis...")
        
        # Step 1: Tokenize and analyze symbol occurrences
        self.symbol_occurrences = self._extract_symbol_occurrences(data)
        print(f"Analyzed {len(self.symbol_occurrences)} symbol occurrences")
        
        # Step 2: Classify semantic contexts
        self._classify_semantic_contexts()
        print(f"Classified {len(set(occ.semantic_context for occ in self.symbol_occurrences))} context types")
        
        # Step 3: Detect multi-meaning symbols
        multi_meaning_symbols = self._detect_multi_meaning_symbols()
        print(f"Detected {len(multi_meaning_symbols)} multi-meaning symbols")
        
        # Step 4: Discover equivalence relationships
        self._discover_equivalence_relationships()
        print(f"Discovered {len(self.equivalence_relations)} equivalence relationships")
        
        # Step 5: Build semantic network
        self._build_semantic_network()
        print(f"Built semantic network with {len(self.semantic_network)} nodes")
        
        # Step 6: Calculate understanding metrics
        understanding_metrics = self._calculate_understanding_metrics(data)
        
        return {
            "multi_meaning_symbols": multi_meaning_symbols,
            "equivalence_relations": [self._relation_to_dict(rel) for rel in self.equivalence_relations],
            "understanding_metrics": understanding_metrics,
            "semantic_network_size": len(self.semantic_network),
            "context_distribution": self._get_context_distribution(),
            "compression_opportunities": self._identify_compression_opportunities()
        }
    
    def _initialize_context_patterns(self) -> List[ContextPattern]:
        """
        Initialize patterns for semantic context classification.
        
        This demonstrates how the system learns to understand different
        types of contextual meanings.
        """
        
        return [
            ContextPattern(
                name="Mathematical Operations",
                regex_patterns=[r'[+\-*/=]', r'\b(equals?|plus|minus|times|divided)\b'],
                keywords=["sum", "product", "result", "calculate", "equation"],
                context_type=SemanticContext.MATHEMATICAL,
                weight=1.0
            ),
            ContextPattern(
                name="Array/Index Operations", 
                regex_patterns=[r'\[.*\]', r'\bindex\b', r'\barray\b'],
                keywords=["element", "position", "index", "array", "list"],
                context_type=SemanticContext.INDEXING,
                weight=0.9
            ),
            ContextPattern(
                name="Procedural Instructions",
                regex_patterns=[r'\b(step|process|execute|run)\b', r'\d+\s*(step|iteration)'],
                keywords=["procedure", "algorithm", "method", "process", "execute"],
                context_type=SemanticContext.PROCEDURAL,
                weight=0.8
            ),
            ContextPattern(
                name="Quantitative Expressions",
                regex_patterns=[r'\b(count|number|quantity|amount)\b', r'\d+\s*(items?|units?)'],
                keywords=["count", "total", "quantity", "amount", "number"],
                context_type=SemanticContext.QUANTITATIVE,
                weight=0.7
            ),
            ContextPattern(
                name="Temporal References",
                regex_patterns=[r'\b(time|second|minute|hour|day)\b', r'\d+\s*(seconds?|minutes?)'],
                keywords=["time", "duration", "period", "interval", "moment"],
                context_type=SemanticContext.TEMPORAL,
                weight=0.8
            ),
            ContextPattern(
                name="Logical Relations",
                regex_patterns=[r'\b(if|then|else|and|or|not)\b', r'\b(true|false)\b'],
                keywords=["condition", "logic", "boolean", "constraint", "rule"],
                context_type=SemanticContext.LOGICAL,
                weight=0.9
            ),
            ContextPattern(
                name="Relational Structures",
                regex_patterns=[r'\b(relationship|connection|link)\b', r'->', r'<->'],
                keywords=["relation", "connection", "association", "link", "reference"],
                context_type=SemanticContext.RELATIONAL,
                weight=0.6
            )
        ]
    
    def _extract_symbol_occurrences(self, data: str) -> List[SymbolOccurrence]:
        """
        Extract all symbol occurrences with their contextual information.
        
        This demonstrates how understanding begins with recognizing symbols
        in their specific contexts.
        """
        
        # Tokenize the data
        tokens = re.findall(r'\w+|\d+|[^\w\s]', data)
        occurrences = []
        
        for i, token in enumerate(tokens):
            # Focus on potentially meaningful symbols (not too short/common)
            if len(token) >= 1 and not self._is_stop_word(token):
                
                # Extract context windows
                context_before = ' '.join(tokens[max(0, i-5):i])
                context_after = ' '.join(tokens[i+1:min(len(tokens), i+6)])
                
                occurrence = SymbolOccurrence(
                    symbol=token,
                    position=i,
                    context_before=context_before,
                    context_after=context_after,
                    semantic_context=SemanticContext.GENERAL,  # Will be classified later
                    confidence=0.0  # Will be calculated later
                )
                
                occurrences.append(occurrence)
        
        return occurrences
    
    def _is_stop_word(self, token: str) -> bool:
        """Check if token is a common stop word."""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did'}
        return token.lower() in stop_words
    
    def _classify_semantic_contexts(self):
        """
        Classify the semantic context of each symbol occurrence.
        
        This is where the system demonstrates UNDERSTANDING by recognizing
        different types of contextual meanings.
        """
        
        for occurrence in self.symbol_occurrences:
            best_context = SemanticContext.GENERAL
            best_confidence = 0.0
            
            # Test against all context patterns
            full_context = f"{occurrence.context_before} {occurrence.symbol} {occurrence.context_after}"
            
            for pattern in self.context_patterns:
                confidence = self._match_context_pattern(full_context, pattern)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_context = pattern.context_type
            
            occurrence.semantic_context = best_context
            occurrence.confidence = best_confidence
    
    def _match_context_pattern(self, context: str, pattern: ContextPattern) -> float:
        """
        Calculate how well a context matches a specific pattern.
        
        This quantifies the system's confidence in understanding context.
        """
        
        context_lower = context.lower()
        score = 0.0
        
        # Check regex patterns
        regex_matches = sum(1 for regex in pattern.regex_patterns 
                           if re.search(regex, context_lower))
        if regex_matches > 0:
            score += 0.4 * (regex_matches / len(pattern.regex_patterns))
        
        # Check keyword matches
        keyword_matches = sum(1 for keyword in pattern.keywords 
                             if keyword in context_lower)
        if keyword_matches > 0:
            score += 0.4 * (keyword_matches / len(pattern.keywords))
        
        # Apply pattern weight
        score *= pattern.weight
        
        return min(1.0, score)
    
    def _detect_multi_meaning_symbols(self) -> Dict[str, List[SemanticContext]]:
        """
        Detect symbols that appear in multiple semantic contexts.
        
        This validates that understanding is required by identifying symbols
        whose meanings depend on context.
        """
        
        symbol_contexts = defaultdict(set)
        
        # Group occurrences by symbol
        for occurrence in self.symbol_occurrences:
            if occurrence.confidence >= self.min_confidence:
                symbol_contexts[occurrence.symbol].add(occurrence.semantic_context)
        
        # Identify multi-meaning symbols
        multi_meaning = {}
        for symbol, contexts in symbol_contexts.items():
            if len(contexts) > 1:
                multi_meaning[symbol] = list(contexts)
        
        return multi_meaning
    
    def _discover_equivalence_relationships(self):
        """
        Discover equivalence relationships between different symbols.
        
        This demonstrates understanding by recognizing when different symbols
        have equivalent meanings in similar contexts.
        """
        
        # Group symbols by context type
        context_groups = defaultdict(list)
        for occurrence in self.symbol_occurrences:
            if occurrence.confidence >= self.min_confidence:
                context_groups[occurrence.semantic_context].append(occurrence)
        
        # Find equivalence relationships within each context
        for context_type, occurrences in context_groups.items():
            symbol_examples = defaultdict(list)
            
            # Collect examples for each symbol in this context
            for occ in occurrences:
                example_context = f"{occ.context_before} <SYMBOL> {occ.context_after}"
                symbol_examples[occ.symbol].append(example_context)
            
            # Find symbols with similar contextual patterns
            symbols = list(symbol_examples.keys())
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol_a, symbol_b = symbols[i], symbols[j]
                    
                    # Calculate equivalence strength based on context similarity
                    equivalence_strength = self._calculate_equivalence_strength(
                        symbol_examples[symbol_a], 
                        symbol_examples[symbol_b]
                    )
                    
                    if equivalence_strength >= self.min_confidence:
                        relation = EquivalenceRelation(
                            symbol_a=symbol_a,
                            symbol_b=symbol_b,
                            context_type=context_type,
                            equivalence_strength=equivalence_strength,
                            examples=list(zip(symbol_examples[symbol_a][:3], 
                                            symbol_examples[symbol_b][:3]))
                        )
                        self.equivalence_relations.append(relation)
    
    def _calculate_equivalence_strength(self, examples_a: List[str], examples_b: List[str]) -> float:
        """
        Calculate the strength of equivalence between two symbol usage patterns.
        
        This quantifies how well the system understands semantic equivalence.
        """
        
        if not examples_a or not examples_b:
            return 0.0
        
        # Simple similarity based on context pattern overlap
        patterns_a = set()
        patterns_b = set()
        
        for example in examples_a:
            patterns_a.update(re.findall(r'\b\w+\b', example.lower()))
        
        for example in examples_b:
            patterns_b.update(re.findall(r'\b\w+\b', example.lower()))
        
        # Calculate Jaccard similarity
        intersection = len(patterns_a & patterns_b)
        union = len(patterns_a | patterns_b)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _build_semantic_network(self):
        """
        Build a network of semantic relationships.
        
        This creates the network-of-information-about-information that
        enables understanding-based storage and retrieval.
        """
        
        # Add equivalence relationships to network
        for relation in self.equivalence_relations:
            self.semantic_network[relation.symbol_a].add(relation.symbol_b)
            self.semantic_network[relation.symbol_b].add(relation.symbol_a)
        
        # Add context-based relationships
        context_symbols = defaultdict(set)
        for occurrence in self.symbol_occurrences:
            context_symbols[occurrence.semantic_context].add(occurrence.symbol)
        
        # Connect symbols that appear in the same contexts
        for context_type, symbols in context_symbols.items():
            symbol_list = list(symbols)
            for i in range(len(symbol_list)):
                for j in range(i + 1, len(symbol_list)):
                    self.semantic_network[symbol_list[i]].add(symbol_list[j])
                    self.semantic_network[symbol_list[j]].add(symbol_list[i])
    
    def _calculate_understanding_metrics(self, data: str) -> Dict[str, float]:
        """
        Calculate quantitative metrics for system understanding.
        
        This validates the equivalence between understanding and optimal processing.
        """
        
        total_symbols = len(self.symbol_occurrences)
        understood_symbols = len([occ for occ in self.symbol_occurrences 
                                if occ.confidence >= self.min_confidence])
        
        # Basic understanding ratio
        understanding_ratio = understood_symbols / total_symbols if total_symbols > 0 else 0
        
        # Context diversity (higher diversity indicates better understanding)
        context_types = set(occ.semantic_context for occ in self.symbol_occurrences)
        context_diversity = len(context_types) / len(SemanticContext)
        
        # Equivalence detection effectiveness
        multi_meaning_symbols = self._detect_multi_meaning_symbols()
        equivalence_effectiveness = len(multi_meaning_symbols) / total_symbols if total_symbols > 0 else 0
        
        # Network connectivity (indicates relationship understanding)
        total_connections = sum(len(connections) for connections in self.semantic_network.values())
        network_density = total_connections / (len(self.semantic_network) ** 2) if len(self.semantic_network) > 1 else 0
        
        return {
            "understanding_ratio": understanding_ratio,
            "context_diversity": context_diversity,
            "equivalence_effectiveness": equivalence_effectiveness,
            "network_density": network_density,
            "total_equivalence_relations": len(self.equivalence_relations)
        }
    
    def _get_context_distribution(self) -> Dict[str, int]:
        """Get distribution of semantic contexts in the data."""
        
        context_counts = Counter(occ.semantic_context.value for occ in self.symbol_occurrences)
        return dict(context_counts)
    
    def _identify_compression_opportunities(self) -> Dict[str, Any]:
        """
        Identify opportunities for compression through equivalence classes.
        
        This validates that understanding enables superior compression.
        """
        
        multi_meaning_symbols = self._detect_multi_meaning_symbols()
        
        # Calculate potential compression savings
        total_occurrences = 0
        compression_savings = 0
        
        for symbol, contexts in multi_meaning_symbols.items():
            symbol_occurrences = [occ for occ in self.symbol_occurrences if occ.symbol == symbol]
            total_occurrences += len(symbol_occurrences)
            
            # Compression savings = occurrences - equivalence class storage - navigation rules
            equivalence_class_cost = 1  # Store one equivalence class
            navigation_rules_cost = len(contexts)  # One rule per context
            
            if len(symbol_occurrences) > (equivalence_class_cost + navigation_rules_cost):
                compression_savings += len(symbol_occurrences) - (equivalence_class_cost + navigation_rules_cost)
        
        compression_ratio = 1 - (compression_savings / total_occurrences) if total_occurrences > 0 else 1
        
        return {
            "multi_meaning_symbol_count": len(multi_meaning_symbols),
            "total_occurrences": total_occurrences,
            "compression_savings": compression_savings,
            "compression_ratio": compression_ratio,
            "top_compression_symbols": sorted(multi_meaning_symbols.items(), 
                                            key=lambda x: len(x[1]), reverse=True)[:5]
        }
    
    def _relation_to_dict(self, relation: EquivalenceRelation) -> Dict[str, Any]:
        """Convert equivalence relation to dictionary for serialization."""
        
        return {
            "symbol_a": relation.symbol_a,
            "symbol_b": relation.symbol_b,
            "context_type": relation.context_type.value,
            "equivalence_strength": relation.equivalence_strength,
            "examples": relation.examples
        }
    
    def export_equivalence_classes(self) -> str:
        """Export equivalence classes as JSON."""
        
        multi_meaning_symbols = self._detect_multi_meaning_symbols()
        
        export_data = {
            "equivalence_classes": multi_meaning_symbols,
            "equivalence_relations": [self._relation_to_dict(rel) for rel in self.equivalence_relations],
            "semantic_network": {symbol: list(connections) for symbol, connections in self.semantic_network.items()},
            "understanding_metrics": self._calculate_understanding_metrics("")
        }
        
        return json.dumps(export_data, indent=2)
    
    def get_symbol_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific symbol."""
        
        symbol_occurrences = [occ for occ in self.symbol_occurrences if occ.symbol == symbol]
        
        if not symbol_occurrences:
            return {"error": f"Symbol '{symbol}' not found in analysis"}
        
        contexts = [occ.semantic_context.value for occ in symbol_occurrences]
        context_distribution = Counter(contexts)
        
        related_symbols = list(self.semantic_network.get(symbol, set()))
        
        return {
            "symbol": symbol,
            "total_occurrences": len(symbol_occurrences),
            "context_distribution": dict(context_distribution),
            "is_multi_meaning": len(set(contexts)) > 1,
            "average_confidence": np.mean([occ.confidence for occ in symbol_occurrences]),
            "related_symbols": related_symbols,
            "example_contexts": [f"{occ.context_before} [{symbol}] {occ.context_after}" 
                               for occ in symbol_occurrences[:3]]
        }
