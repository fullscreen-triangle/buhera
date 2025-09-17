#!/usr/bin/env python3
"""
Example Usage of Buhera Framework Validation Package

This script demonstrates how to use the validation package to verify
the core principles of the Buhera VPOS framework.
"""

import os
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from buhera_validation import (
    MetaInformationCascade,
    EquivalenceDetector,
    UnderstandingNetwork,
    CompressionDemo,
    NetworkEvolutionDemo
)


def quick_compression_test():
    """Quick test of the meta-information cascade compression."""
    
    print("=== Quick Compression Test ===")
    
    # Test data with multi-meaning symbols
    test_data = """
    The number 5 represents different concepts:
    Mathematical: 2 + 3 = 5
    Array index: element[5] 
    Iteration count: repeat 5 times
    Each context gives 5 different meaning.
    """
    
    # Initialize compressor
    compressor = MetaInformationCascade()
    
    # Compress and analyze
    result = compressor.compress(test_data)
    
    print(f"Original size: {result.original_size} bytes")
    print(f"Compressed size: {result.cascade_size} bytes")
    print(f"Compression ratio: {result.compression_ratio:.3f}")
    print(f"Understanding score: {result.understanding_score:.3f}")
    print(f"Equivalence classes found: {len(result.equivalence_classes)}")
    print(f"Navigation rules created: {len(result.navigation_rules)}")
    
    print("✓ Compression test complete\n")


def quick_equivalence_test():
    """Quick test of equivalence class detection."""
    
    print("=== Quick Equivalence Detection Test ===")
    
    # Test data for equivalence detection
    test_data = """
    Process the data using algorithm A.
    The method A provides optimal results.
    Technique A minimizes overhead.
    Algorithm A complexity is O(n).
    """
    
    # Initialize detector
    detector = EquivalenceDetector()
    
    # Analyze equivalence classes
    analysis = detector.analyze_data(test_data)
    
    print(f"Multi-meaning symbols detected: {len(analysis['multi_meaning_symbols'])}")
    print(f"Context types identified: {len(analysis['context_distribution'])}")
    print(f"Understanding ratio: {analysis['understanding_metrics']['understanding_ratio']:.3f}")
    print(f"Equivalence relations: {analysis['understanding_metrics']['total_equivalence_relations']}")
    
    # Show specific symbol analysis
    if analysis['multi_meaning_symbols']:
        symbol = list(analysis['multi_meaning_symbols'].keys())[0]
        symbol_analysis = detector.get_symbol_analysis(symbol)
        print(f"\nExample symbol '{symbol}':")
        print(f"  Total occurrences: {symbol_analysis['total_occurrences']}")
        print(f"  Is multi-meaning: {symbol_analysis['is_multi_meaning']}")
        print(f"  Context distribution: {symbol_analysis['context_distribution']}")
    
    print("✓ Equivalence detection test complete\n")


def quick_network_test():
    """Quick test of understanding network evolution."""
    
    print("=== Quick Network Evolution Test ===")
    
    # Initialize network
    network = UnderstandingNetwork()
    
    # Test sequence of information
    info_sequence = [
        "The symbol X has multiple meanings.",
        "X represents variable in mathematics.",
        "X marks location in navigation.",
        "Understanding X requires context analysis."
    ]
    
    # Ingest information and track evolution
    results = []
    for i, info in enumerate(info_sequence):
        result = network.ingest_information(info)
        results.append(result)
        print(f"  Step {i+1}: Added '{info[:30]}...', Network size: {len(network.information_nodes)}")
    
    # Final network state
    print(f"\nFinal network state:")
    print(f"  Total nodes: {len(network.information_nodes)}")
    print(f"  Understanding patterns: {len(network.understanding_patterns)}")
    print(f"  Evolution events: {len(network.evolution_history)}")
    
    print("✓ Network evolution test complete\n")


def run_simple_demo():
    """Run a simple demonstration of the core framework components."""
    
    print("=" * 60)
    print("BUHERA FRAMEWORK VALIDATION - SIMPLE DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Run quick tests
    quick_compression_test()
    quick_equivalence_test()
    quick_network_test()
    
    print("=" * 60)
    print("SIMPLE DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("To run comprehensive validation:")
    print("  python -m buhera_validation.cli --full-suite")
    print()
    print("To run specific validations:")
    print("  python -m buhera_validation.cli --compression")
    print("  python -m buhera_validation.cli --network-evolution")


def run_full_demonstrations():
    """Run the full demonstration suite."""
    
    print("=" * 70)
    print("BUHERA FRAMEWORK VALIDATION - FULL DEMONSTRATION SUITE")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Run compression demonstration
    print("Phase 1: Compression Algorithm Validation")
    print("-" * 50)
    compression_demo = CompressionDemo()
    compression_results = compression_demo.run_full_validation()
    print("✓ Compression validation complete\n")
    
    # Run network evolution demonstration
    print("Phase 2: Network Understanding Evolution")
    print("-" * 50)
    network_demo = NetworkEvolutionDemo()
    network_results = network_demo.demonstrate_understanding_accumulation()
    print("✓ Network evolution validation complete\n")
    
    # Summary
    print("Phase 3: Final Summary")
    print("-" * 50)
    
    # Extract key results
    compression_validated = compression_results["validation_summary"]["validation_status"]["framework_validated"]
    network_validated = network_results["learning_analysis"]["validation_success"]
    
    print(f"Compression Algorithm Validation: {'✓ PASSED' if compression_validated else '✗ FAILED'}")
    print(f"Network Evolution Validation: {'✓ PASSED' if network_validated else '✗ FAILED'}")
    
    overall_success = compression_validated and network_validated
    print(f"\nOVERALL FRAMEWORK VALIDATION: {'✓ SUCCESS' if overall_success else '✗ INCOMPLETE'}")
    
    if overall_success:
        print("\n🎉 BUHERA FRAMEWORK SUCCESSFULLY VALIDATED! 🎉")
        print("Core principles confirmed through measurable experiments:")
        print("  • Storage = Understanding equivalence proven")
        print("  • Meta-information cascade compression validated")
        print("  • Understanding network accumulation demonstrated")
        print("  • Navigation-based retrieval confirmed")
    else:
        print("\n⚠️ Framework validation incomplete. Review component results.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    """Main execution."""
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_demonstrations()
    else:
        run_simple_demo()
        
        print("To run full demonstrations with comprehensive validation:")
        print(f"  python {sys.argv[0]} --full")
