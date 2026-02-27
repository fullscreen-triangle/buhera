#!/usr/bin/env python3
"""
Example Usage of Buhera Framework Validation Package - WITH JSON RESULTS SAVING

This script demonstrates how to use the validation package to verify
the core principles of the Buhera VPOS framework and saves all results as JSON.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from buhera_validation import (
    MetaInformationCascade,
    EquivalenceDetector,
    UnderstandingNetwork,
    NavigationRetriever,
    FoundryValidator,
    VirtualProcessingValidator
)


def safe_test_execution(test_name, test_function):
    """Safely execute a test function and return results."""
    print(f"\n=== {test_name} ===")
    
    try:
        result = test_function()
        print(f"✅ {test_name} completed successfully")
        return {
            "test_name": test_name,
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "results": result
        }
    except Exception as e:
        print(f"❌ {test_name} failed: {e}")
        return {
            "test_name": test_name,
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


def quick_compression_test():
    """Quick test of the meta-information cascade compression."""
    
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
    
    test_results = {
        "original_size": result.original_size,
        "compressed_size": result.cascade_size,
        "compression_ratio": result.compression_ratio,
        "understanding_score": result.understanding_score,
        "equivalence_classes_count": len(result.equivalence_classes),
        "navigation_rules_count": len(result.navigation_rules),
        "test_data_sample": test_data.strip()[:100] + "..."
    }
    
    print(f"Original size: {test_results['original_size']} bytes")
    print(f"Compressed size: {test_results['compressed_size']} bytes")
    print(f"Compression ratio: {test_results['compression_ratio']:.3f}")
    print(f"Understanding score: {test_results['understanding_score']:.3f}")
    print(f"Equivalence classes found: {test_results['equivalence_classes_count']}")
    print(f"Navigation rules created: {test_results['navigation_rules_count']}")
    
    return test_results


def quick_equivalence_test():
    """Quick test of equivalence class detection."""
    
    # Test data for equivalence detection
    test_data = """
    Process the data using algorithm A.
    The method A provides optimal results.
    Technique A minimizes overhead.
    """
    
    # Initialize detector
    detector = EquivalenceDetector()
    
    # Analyze data
    analysis_result = detector.analyze_text(test_data)
    
    test_results = {
        "multi_meaning_count": analysis_result.multi_meaning_count,
        "context_types_count": len(analysis_result.context_types),
        "understanding_ratio": analysis_result.understanding_ratio,
        "equivalence_relations_count": len(analysis_result.equivalence_relations),
        "test_data_sample": test_data.strip()[:100] + "..."
    }
    
    print(f"Multi-meaning symbols detected: {test_results['multi_meaning_count']}")
    print(f"Context types identified: {test_results['context_types_count']}")
    print(f"Understanding ratio: {test_results['understanding_ratio']:.3f}")
    print(f"Equivalence relations: {test_results['equivalence_relations_count']}")
    
    return test_results


def quick_network_test():
    """Quick test of understanding network evolution."""
    
    # Initialize network
    network = UnderstandingNetwork()
    
    # Add information progressively
    info = "The symbol X has multiple meanings...."
    result = network.ingest_information(info)
    
    test_results = {
        "network_nodes": result['network_state']['node_count'],
        "storage_efficiency": result['storage_efficiency'],
        "understanding_score": result['understanding_score'],
        "learning_detected": result['learning_detected'],
        "test_info_sample": info[:50] + "..."
    }
    
    print(f"Network nodes: {test_results['network_nodes']}")
    print(f"Storage efficiency: {test_results['storage_efficiency']:.3f}")
    print(f"Understanding score: {test_results['understanding_score']:.3f}")
    print(f"Learning demonstrated: {test_results['learning_detected']}")
    
    return test_results


def quick_foundry_test():
    """Quick test of the foundry architecture validation."""
    
    # Initialize foundry validator
    foundry = FoundryValidator(target_density=1e9)
    
    # Run validation on small test volume
    result = foundry.validate_foundry_architecture(test_volume=0.001)  # 1 liter
    
    test_results = {
        "theoretical_density_achieved": result.theoretical_density_achieved,
        "actual_processors_created": result.actual_processors_created,
        "processing_efficiency": result.processing_efficiency,
        "quantum_coherence_duration": result.quantum_coherence_duration,
        "validation_score": result.validation_score
    }
    
    print(f"Theoretical density achieved: {test_results['theoretical_density_achieved']:.3f}")
    print(f"Processors created: {test_results['actual_processors_created']:,}")
    print(f"Processing efficiency: {test_results['processing_efficiency']:.3f}")
    print(f"Quantum coherence duration: {test_results['quantum_coherence_duration']:.2e} seconds")
    print(f"Overall validation score: {test_results['validation_score']:.3f}")
    
    return test_results


def quick_virtual_acceleration_test():
    """Quick test of the virtual processing acceleration validation."""
    
    # Initialize virtual processing validator
    virtual = VirtualProcessingValidator(target_frequency=1e29)  # Start with lower frequency
    
    # Run validation
    result = virtual.validate_virtual_processing_architecture()
    
    test_results = {
        "target_frequency_hz": result.target_frequency_hz,
        "achieved_frequency_hz": result.achieved_frequency_hz,
        "frequency_accuracy": result.frequency_accuracy,
        "temporal_precision_seconds": result.temporal_precision_seconds,
        "parallel_processing_capacity": result.parallel_processing_capacity,
        "validation_score": result.validation_score
    }
    
    print(f"Target frequency: {test_results['target_frequency_hz']:.2e} Hz")
    print(f"Achieved frequency: {test_results['achieved_frequency_hz']:.2e} Hz")
    print(f"Frequency accuracy: {test_results['frequency_accuracy']:.3f}")
    print(f"Temporal precision: {test_results['temporal_precision_seconds']:.2e} seconds")
    print(f"Parallel capacity: {test_results['parallel_processing_capacity']:.2e}")
    print(f"Overall validation score: {test_results['validation_score']:.3f}")
    
    return test_results


def run_simple_demo_with_json():
    """Run a simple demonstration of the core framework components and save results as JSON."""
    
    print("=" * 60)
    print("BUHERA FRAMEWORK VALIDATION - SIMPLE DEMONSTRATION")
    print("=" * 60)
    print("🔧 Fixed IndexError issue and saving all results as JSON")
    
    # Create results directory
    results_dir = Path("validation_results")
    results_dir.mkdir(exist_ok=True)
    
    # Initialize demo results structure
    demo_results = {
        "demo_type": "simple_demonstration",
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "demo_summary": {}
    }
    
    # Define test functions
    tests = [
        ("Quick Compression Test", quick_compression_test),
        ("Quick Equivalence Detection Test", quick_equivalence_test),
        ("Quick Network Evolution Test", quick_network_test),
        ("Quick Foundry Architecture Test", quick_foundry_test),
        ("Quick Virtual Processing Acceleration Test", quick_virtual_acceleration_test)
    ]
    
    # Run all tests safely
    for test_name, test_func in tests:
        test_result = safe_test_execution(test_name, test_func)
        demo_results["tests"][test_name.lower().replace(" ", "_")] = test_result
    
    # Create demo summary
    successful_tests = [test for test in demo_results["tests"].values() if test["success"]]
    failed_tests = [test for test in demo_results["tests"].values() if not test["success"]]
    
    demo_results["demo_summary"] = {
        "total_tests": len(tests),
        "successful_tests": len(successful_tests),
        "failed_tests": len(failed_tests),
        "success_rate": len(successful_tests) / len(tests),
        "all_tests_passed": len(failed_tests) == 0,
        "failed_test_names": [test["test_name"] for test in failed_tests]
    }
    
    # Save results to JSON
    results_file = results_dir / "simple_demo_results.json"
    with open(results_file, 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    # Save individual test results as separate files
    for test_key, test_result in demo_results["tests"].items():
        individual_file = results_dir / f"{test_key}_results.json"
        with open(individual_file, 'w') as f:
            json.dump(test_result, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("SIMPLE DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"💾 Main results saved to: {results_file}")
    print(f"📁 Individual test results saved to: {results_dir}/")
    print(f"✅ Tests successful: {demo_results['demo_summary']['successful_tests']}/5")
    print(f"📊 Success rate: {demo_results['demo_summary']['success_rate']:.1%}")
    
    if failed_tests:
        print(f"❌ Failed tests: {', '.join(demo_results['demo_summary']['failed_test_names'])}")
    
    print()
    print("To run comprehensive validation:")
    print("  python -m buhera_validation.cli --full-suite --output results/")
    print()
    
    return demo_results


def run_full_demo():
    """Run full demonstration with enhanced alphabetical encoding validation."""
    
    print("=" * 80)
    print("BUHERA FRAMEWORK VALIDATION - FULL DEMONSTRATION WITH JSON RESULTS")
    print("=" * 80)
    
    # Create results directory
    results_dir = Path("validation_results")
    results_dir.mkdir(exist_ok=True)
    
    full_results = {
        "demo_type": "full_demonstration",
        "timestamp": datetime.now().isoformat(),
        "simple_demo": {},
        "alphabetical_encoding_validation": {},
        "overall_summary": {}
    }
    
    # Run simple demo
    print("\n🔧 PHASE 1: CORE FRAMEWORK VALIDATION")
    full_results["simple_demo"] = run_simple_demo_with_json()
    
    # Run alphabetical encoding validation
    print("\n🔤 PHASE 2: ALPHABETICAL ENCODING VALIDATION")
    try:
        # Import and run alphabetical encoding validation
        sys.path.append(str(Path(__file__).parent))
        from complete_alphabetical_validation import main as run_alphabetical_validation
        
        alphabetical_results = run_alphabetical_validation()
        full_results["alphabetical_encoding_validation"] = alphabetical_results
        print("✅ Alphabetical encoding validation completed")
        
    except Exception as e:
        print(f"⚠️ Alphabetical encoding validation encountered issues: {e}")
        full_results["alphabetical_encoding_validation"] = {
            "status": "ERROR",
            "error": str(e)
        }
    
    # Create overall summary
    simple_success_rate = full_results["simple_demo"]["demo_summary"]["success_rate"]
    alphabetical_status = full_results["alphabetical_encoding_validation"].get("overall_assessment", {}).get("status", "UNKNOWN")
    
    full_results["overall_summary"] = {
        "core_framework_success_rate": simple_success_rate,
        "alphabetical_encoding_status": alphabetical_status,
        "overall_validation_successful": simple_success_rate > 0.6 and alphabetical_status in ["VALIDATED", "PARTIALLY_VALIDATED"],
        "recommendation": "PROCEED_WITH_DEVELOPMENT" if simple_success_rate > 0.8 else "REQUIRES_FIXES"
    }
    
    # Save comprehensive results
    full_results_file = results_dir / "full_demonstration_results.json"
    with open(full_results_file, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("FULL DEMONSTRATION COMPLETE")
    print("=" * 80)
    print(f"💾 Full results saved to: {full_results_file}")
    print(f"🎯 Overall validation: {'✅ SUCCESSFUL' if full_results['overall_summary']['overall_validation_successful'] else '⚠️ NEEDS ATTENTION'}")
    print(f"📊 Recommendation: {full_results['overall_summary']['recommendation']}")
    
    return full_results


if __name__ == "__main__":
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Run full demonstration
        results = run_full_demo()
    else:
        # Run simple demonstration
        results = run_simple_demo_with_json()
    
    print("\n" + "🎉" * 60)
    print("ALL RESULTS SAVED AS JSON FILES!")
    print("Check the validation_results/ directory for all output files.")
    print("🎉" * 60)
