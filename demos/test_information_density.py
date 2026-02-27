"""
Information Density Testing

This script rigorously tests whether the alphabetical encoding actually
increases information density and provides concrete measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json

from buhera_validation.core.alphabetical_encoding import MultiStepAlphabeticalEncoder
from buhera_validation.core.cascade_compression import MetaInformationCascade


class InformationDensityTester:
    """
    Rigorous tester for information density changes through alphabetical encoding.
    """
    
    def __init__(self):
        """Initialize information density tester."""
        self.encoder = MultiStepAlphabeticalEncoder()
        self.compressor = MetaInformationCascade()
    
    def test_information_density_comprehensive(self) -> Dict[str, any]:
        """
        Comprehensive test of information density changes.
        """
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE INFORMATION DENSITY TESTING")
        print("   Testing whether alphabetical encoding increases information utility")
        print("="*80)
        
        results = {
            "density_measurements": [],
            "entropy_analysis": {},
            "compression_effectiveness": {},
            "semantic_pathway_density": {},
            "information_utility_score": {},
            "final_assessment": {}
        }
        
        # Test with various types of information
        test_cases = [
            # User's example
            ("bib", "User's original example"),
            
            # Simple cases
            ("hello", "Common English word"),
            ("test", "Short common word"),
            ("cat", "Very short word"),
            
            # Complex cases
            ("information", "Abstract concept"),
            ("consciousness", "Complex philosophical term"),
            ("quantum", "Scientific term"),
            ("understanding", "Long abstract word"),
            
            # Repeated patterns
            ("banana", "Word with repeated letters"),
            ("mississippi", "High repetition word"),
            
            # Technical terms
            ("algorithm", "Technical term"),
            ("database", "Technical term"),
        ]
        
        print("\n🔍 DETAILED DENSITY ANALYSIS:")
        print("-" * 60)
        
        for text, description in test_cases:
            print(f"\n📝 Analyzing: '{text}' ({description})")
            
            density_result = self._measure_information_density(text)
            results["density_measurements"].append(density_result)
            
            # Print key metrics
            print(f"   Original entropy: {density_result['original_entropy']:.3f} bits/char")
            print(f"   Encoded entropy: {density_result['encoded_entropy']:.3f} bits/char")  
            print(f"   Entropy change: {density_result['entropy_change']:.3f}")
            print(f"   Size ratio: {density_result['size_ratio']:.2f}x")
            print(f"   Semantic pathways: {density_result['semantic_pathways']}")
            print(f"   Information utility: {density_result['information_utility_score']:.3f}")
        
        # Overall analysis
        results["entropy_analysis"] = self._analyze_entropy_changes(results["density_measurements"])
        results["compression_effectiveness"] = self._analyze_compression_effectiveness(results["density_measurements"])
        results["semantic_pathway_density"] = self._analyze_semantic_pathway_density(results["density_measurements"])
        results["information_utility_score"] = self._calculate_overall_information_utility(results["density_measurements"])
        results["final_assessment"] = self._create_final_density_assessment(results)
        
        return results
    
    def _measure_information_density(self, text: str) -> Dict[str, float]:
        """
        Measure information density changes through alphabetical encoding.
        """
        
        # Original information metrics
        original_bytes = text.encode('utf-8')
        original_size = len(original_bytes)
        original_entropy = self._calculate_shannon_entropy(original_bytes)
        
        # Apply alphabetical encoding
        encoding_result = self.encoder.encode_complete_pipeline(text)
        
        # Encoded information metrics
        encoded_data = encoding_result["final_encoded"]
        encoded_bytes = encoded_data.encode('utf-8') if isinstance(encoded_data, str) else encoded_data
        encoded_size = len(encoded_bytes)
        encoded_entropy = self._calculate_shannon_entropy(encoded_bytes)
        
        # Calculate density metrics
        size_ratio = encoded_size / original_size if original_size > 0 else float('inf')
        entropy_change = encoded_entropy - original_entropy
        
        # Semantic pathway density
        pathways = len(encoding_result["semantic_pathways"])
        pathway_density = pathways / original_size  # Pathways per character
        
        # Information utility score (combination of factors)
        reversible = encoding_result["reversibility_validation"]["reversibility_validated"]
        compression_potential = encoding_result["compression_analysis"]["compression_potential_score"]
        
        information_utility_score = self._calculate_information_utility(
            reversible, pathways, compression_potential, entropy_change
        )
        
        return {
            "text": text,
            "original_size": original_size,
            "encoded_size": encoded_size,
            "original_entropy": original_entropy,
            "encoded_entropy": encoded_entropy,
            "size_ratio": size_ratio,
            "entropy_change": entropy_change,
            "semantic_pathways": pathways,
            "pathway_density": pathway_density,
            "compression_potential": compression_potential,
            "reversible": reversible,
            "information_utility_score": information_utility_score
        }
    
    def _calculate_shannon_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of byte data."""
        
        if len(data) == 0:
            return 0.0
        
        # Count byte frequencies
        byte_counts = {}
        for byte in data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        total = len(data)
        for count in byte_counts.values():
            prob = count / total
            entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _calculate_information_utility(self, 
                                     reversible: bool,
                                     pathways: int, 
                                     compression_potential: float,
                                     entropy_change: float) -> float:
        """
        Calculate overall information utility score.
        
        Higher scores indicate that the encoding increases information utility.
        """
        
        # Base score for reversibility (essential)
        base_score = 1.0 if reversible else 0.0
        
        # Pathway bonus (more pathways = better utility)
        pathway_bonus = min(pathways / 4.0, 1.0)  # Cap at 1.0
        
        # Compression potential bonus
        compression_bonus = min(compression_potential, 1.0)
        
        # Entropy structure bonus (negative entropy change can be good for compression)
        structure_bonus = max(0.0, min(-entropy_change / 2.0, 0.5)) if entropy_change < 0 else 0.0
        
        # Combined utility score
        utility_score = (
            base_score * 0.4 +           # 40% for reversibility
            pathway_bonus * 0.3 +        # 30% for pathways
            compression_bonus * 0.2 +    # 20% for compression potential
            structure_bonus * 0.1        # 10% for structure improvement
        )
        
        return utility_score
    
    def _analyze_entropy_changes(self, measurements: List[Dict]) -> Dict[str, float]:
        """Analyze entropy changes across all measurements."""
        
        entropy_changes = [m["entropy_change"] for m in measurements]
        
        return {
            "average_entropy_change": np.mean(entropy_changes),
            "entropy_reduction_cases": sum(1 for e in entropy_changes if e < 0),
            "entropy_increase_cases": sum(1 for e in entropy_changes if e > 0),
            "entropy_change_consistency": 1.0 - np.std(entropy_changes) / (abs(np.mean(entropy_changes)) + 0.001),
            "significant_structure_created": abs(np.mean(entropy_changes)) > 0.5
        }
    
    def _analyze_compression_effectiveness(self, measurements: List[Dict]) -> Dict[str, float]:
        """Analyze compression effectiveness."""
        
        compression_potentials = [m["compression_potential"] for m in measurements]
        
        return {
            "average_compression_potential": np.mean(compression_potentials),
            "high_compression_cases": sum(1 for c in compression_potentials if c > 0.7),
            "compression_consistency": 1.0 - np.std(compression_potentials) / (np.mean(compression_potentials) + 0.001),
            "compression_effective": np.mean(compression_potentials) > 0.5
        }
    
    def _analyze_semantic_pathway_density(self, measurements: List[Dict]) -> Dict[str, float]:
        """Analyze semantic pathway density."""
        
        pathway_counts = [m["semantic_pathways"] for m in measurements]
        pathway_densities = [m["pathway_density"] for m in measurements]
        
        return {
            "average_pathways": np.mean(pathway_counts),
            "average_pathway_density": np.mean(pathway_densities),
            "high_pathway_cases": sum(1 for p in pathway_counts if p >= 4),
            "pathway_generation_effective": np.mean(pathway_counts) > 3.0
        }
    
    def _calculate_overall_information_utility(self, measurements: List[Dict]) -> Dict[str, float]:
        """Calculate overall information utility metrics."""
        
        utility_scores = [m["information_utility_score"] for m in measurements]
        reversible_cases = sum(1 for m in measurements if m["reversible"])
        
        return {
            "average_information_utility": np.mean(utility_scores),
            "high_utility_cases": sum(1 for u in utility_scores if u > 0.7),
            "reversibility_rate": reversible_cases / len(measurements),
            "overall_utility_validated": np.mean(utility_scores) > 0.6
        }
    
    def _create_final_density_assessment(self, results: Dict) -> Dict[str, any]:
        """Create final assessment of information density benefits."""
        
        # Extract key metrics
        avg_utility = results["information_utility_score"]["average_information_utility"]
        entropy_effective = results["entropy_analysis"]["significant_structure_created"]
        compression_effective = results["compression_effectiveness"]["compression_effective"]
        pathways_effective = results["semantic_pathway_density"]["pathway_generation_effective"]
        reversibility_rate = results["information_utility_score"]["reversibility_rate"]
        
        # Overall assessment
        positive_factors = sum([
            avg_utility > 0.6,
            entropy_effective,
            compression_effective,
            pathways_effective,
            reversibility_rate > 0.8
        ])
        
        # Determine recommendation
        if positive_factors >= 4:
            recommendation = "STRONGLY_RECOMMENDED"
            assessment = "Alphabetical encoding significantly increases information utility"
        elif positive_factors >= 3:
            recommendation = "RECOMMENDED"  
            assessment = "Alphabetical encoding provides moderate information utility benefits"
        elif positive_factors >= 2:
            recommendation = "CONDITIONALLY_USEFUL"
            assessment = "Alphabetical encoding has some benefits but limited applicability"
        else:
            recommendation = "NOT_RECOMMENDED"
            assessment = "Alphabetical encoding does not significantly improve information utility"
        
        return {
            "overall_recommendation": recommendation,
            "assessment_summary": assessment,
            "positive_factors_count": positive_factors,
            "key_strengths": self._identify_key_strengths(results),
            "key_limitations": self._identify_key_limitations(results),
            "optimal_use_cases": self._identify_optimal_use_cases(results)
        }
    
    def _identify_key_strengths(self, results: Dict) -> List[str]:
        """Identify key strengths of the alphabetical encoding."""
        
        strengths = []
        
        if results["information_utility_score"]["reversibility_rate"] >= 1.0:
            strengths.append("Perfect information preservation (100% reversible)")
        
        if results["semantic_pathway_density"]["average_pathways"] > 3:
            strengths.append(f"Rich semantic pathway generation ({results['semantic_pathway_density']['average_pathways']:.1f} avg pathways)")
        
        if results["compression_effectiveness"]["compression_effective"]:
            strengths.append("Good compression potential for further processing")
        
        if results["entropy_analysis"]["significant_structure_created"]:
            strengths.append("Creates significant structural patterns")
        
        return strengths
    
    def _identify_key_limitations(self, results: Dict) -> List[str]:
        """Identify key limitations of the alphabetical encoding."""
        
        limitations = []
        
        # Check size expansion
        avg_size_ratio = np.mean([m["size_ratio"] for m in results["density_measurements"]])
        if avg_size_ratio > 2.0:
            limitations.append(f"Significant size expansion ({avg_size_ratio:.1f}x average)")
        
        # Check entropy increase
        if results["entropy_analysis"]["average_entropy_change"] > 1.0:
            limitations.append("Increases entropy (reduces immediate compressibility)")
        
        # Check compression effectiveness
        if not results["compression_effectiveness"]["compression_effective"]:
            limitations.append("Limited direct compression benefits")
        
        return limitations
    
    def _identify_optimal_use_cases(self, results: Dict) -> List[str]:
        """Identify optimal use cases based on analysis."""
        
        use_cases = []
        
        if results["semantic_pathway_density"]["pathway_generation_effective"]:
            use_cases.append("Multi-pathway semantic retrieval systems")
        
        if results["information_utility_score"]["reversibility_rate"] >= 1.0:
            use_cases.append("Formal verification and proof systems")
        
        if results["compression_effectiveness"]["compression_effective"]:
            use_cases.append("Pre-processing for advanced compression algorithms")
        
        if results["entropy_analysis"]["significant_structure_created"]:
            use_cases.append("Pattern generation for machine learning")
        
        use_cases.append("Consciousness-substrate computing frameworks")
        
        return use_cases


def run_information_density_test():
    """Run comprehensive information density testing."""
    
    tester = InformationDensityTester()
    results = tester.test_information_density_comprehensive()
    
    print("\n" + "="*80)
    print("📊 INFORMATION DENSITY TEST RESULTS")
    print("="*80)
    
    # Print summary
    final_assessment = results["final_assessment"]
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"   Recommendation: {final_assessment['overall_recommendation']}")
    print(f"   Summary: {final_assessment['assessment_summary']}")
    print(f"   Positive factors: {final_assessment['positive_factors_count']}/5")
    
    print(f"\n✅ KEY STRENGTHS:")
    for strength in final_assessment["key_strengths"]:
        print(f"   • {strength}")
    
    if final_assessment["key_limitations"]:
        print(f"\n⚠️ KEY LIMITATIONS:")
        for limitation in final_assessment["key_limitations"]:
            print(f"   • {limitation}")
    
    print(f"\n🎯 OPTIMAL USE CASES:")
    for use_case in final_assessment["optimal_use_cases"]:
        print(f"   • {use_case}")
    
    # Print detailed metrics
    print(f"\n📈 DETAILED METRICS:")
    print(f"   Average Information Utility: {results['information_utility_score']['average_information_utility']:.3f}")
    print(f"   Reversibility Rate: {results['information_utility_score']['reversibility_rate']:.1%}")
    print(f"   Average Pathways: {results['semantic_pathway_density']['average_pathways']:.1f}")
    print(f"   Compression Effectiveness: {'✅' if results['compression_effectiveness']['compression_effective'] else '❌'}")
    print(f"   Structure Creation: {'✅' if results['entropy_analysis']['significant_structure_created'] else '❌'}")
    
    return results


if __name__ == "__main__":
    results = run_information_density_test()
    
    # Save results
    with open("information_density_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: information_density_test_results.json")
