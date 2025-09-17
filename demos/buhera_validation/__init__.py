"""
Buhera Framework Validation Package

A comprehensive validation suite for the revolutionary Buhera VPOS 
consciousness-substrate computing framework.

This package demonstrates that:
1. Storage and understanding are mathematically equivalent
2. Optimal compression requires semantic comprehension  
3. Information networks evolve through accumulated understanding
4. Navigation-based retrieval eliminates traditional computation

Each module provides working implementations with measurable validation.
"""

__version__ = "1.0.0"
__author__ = "Buhera Research Team"

# Core framework components
from .core.cascade_compression import MetaInformationCascade
from .core.equivalence_detection import EquivalenceDetector
from .core.navigation_retrieval import NavigationRetriever
from .core.network_understanding import UnderstandingNetwork
from .core.foundry_validation import FoundryValidator
from .core.virtual_processing_validation import VirtualProcessingValidator
from .core.proof_validated_storage import ProofValidatedCascadeStorage, ProofSystem, ProofTerm

# Validation demonstrations
from .demonstrations.compression_demo import CompressionDemo
from .demonstrations.understanding_demo import UnderstandingDemo
from .demonstrations.network_evolution_demo import NetworkEvolutionDemo
from .demonstrations.foundry_demo import FoundryDemo
from .demonstrations.virtual_acceleration_demo import VirtualAccelerationDemo
from .demonstrations.proof_validated_demo import ProofValidatedStorageDemo

# Comprehensive managers
from .visualization_manager import BuheraVisualizationManager
from .results_manager import BuheraResultsManager

__all__ = [
    # Core components
    'MetaInformationCascade',
    'EquivalenceDetector', 
    'NavigationRetriever',
    'UnderstandingNetwork',
    'FoundryValidator',
    'VirtualProcessingValidator',
    'ProofValidatedCascadeStorage',
    'ProofSystem',
    'ProofTerm',
    
    # Demonstrations
    'CompressionDemo',
    'UnderstandingDemo',
    'NetworkEvolutionDemo',
    'FoundryDemo',
    'VirtualAccelerationDemo',
    'ProofValidatedStorageDemo',
    
    # Comprehensive managers
    'BuheraVisualizationManager',
    'BuheraResultsManager',
]
