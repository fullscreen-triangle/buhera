"""
Comprehensive Visualization Manager

This module generates extensive visualizations for all aspects of the
Buhera framework validation, creating publication-ready figures and
comprehensive analysis charts.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BuheraVisualizationManager:
    """
    Comprehensive visualization manager for Buhera framework validation.
    
    Generates extensive publication-ready visualizations including:
    - Performance dashboards
    - Detailed component analysis
    - Comparative studies
    - Interactive plots
    - Summary reports
    """
    
    def __init__(self, output_dir: str = "validation_results"):
        """Initialize visualization manager."""
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Publication-quality settings
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
    
    def create_comprehensive_dashboard(self, 
                                     compression_results: Dict[str, Any],
                                     network_results: Dict[str, Any],
                                     foundry_results: Optional[Dict[str, Any]] = None,
                                     virtual_results: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Create comprehensive validation dashboard with all key metrics.
        """
        
        print("Creating comprehensive validation dashboard...")
        
        generated_files = []
        
        # Dashboard 1: Executive Summary Dashboard
        fig_path = self._create_executive_summary_dashboard(compression_results, network_results, foundry_results, virtual_results)
        generated_files.append(fig_path)
        
        # Dashboard 2: Detailed Performance Metrics
        fig_path = self._create_detailed_performance_dashboard(compression_results, network_results, foundry_results, virtual_results)
        generated_files.append(fig_path)
        
        # Dashboard 3: Component Analysis Dashboard
        fig_path = self._create_component_analysis_dashboard(compression_results, network_results, foundry_results, virtual_results)
        generated_files.append(fig_path)
        
        # Dashboard 4: Validation Status Dashboard
        fig_path = self._create_validation_status_dashboard(compression_results, network_results, foundry_results, virtual_results)
        generated_files.append(fig_path)
        
        return generated_files
    
    def create_compression_visualizations(self, results: Dict[str, Any]) -> List[str]:
        """Create comprehensive compression validation visualizations."""
        
        print("Creating compression validation visualizations...")
        generated_files = []
        
        # Extract data
        compression_data = results["compression_comparisons"]
        
        # Figure 1: Compression Performance Comparison
        fig_path = self._create_compression_performance_chart(compression_data)
        generated_files.append(fig_path)
        
        # Figure 2: Understanding vs Compression Analysis
        fig_path = self._create_understanding_compression_analysis(compression_data)
        generated_files.append(fig_path)
        
        # Figure 3: Context Detection Analysis
        fig_path = self._create_context_detection_analysis(results)
        generated_files.append(fig_path)
        
        # Figure 4: Navigation Rules Effectiveness
        fig_path = self._create_navigation_rules_analysis(results)
        generated_files.append(fig_path)
        
        return generated_files
    
    def create_network_visualizations(self, results: Dict[str, Any]) -> List[str]:
        """Create comprehensive network evolution visualizations."""
        
        print("Creating network evolution visualizations...")
        generated_files = []
        
        # Figure 1: Learning Progression Analysis
        fig_path = self._create_learning_progression_chart(results)
        generated_files.append(fig_path)
        
        # Figure 2: Network Growth Dynamics
        fig_path = self._create_network_growth_dynamics(results)
        generated_files.append(fig_path)
        
        # Figure 3: Understanding Accumulation Heatmap
        fig_path = self._create_understanding_accumulation_heatmap(results)
        generated_files.append(fig_path)
        
        # Figure 4: Storage Pattern Evolution
        fig_path = self._create_storage_pattern_evolution(results)
        generated_files.append(fig_path)
        
        return generated_files
    
    def create_foundry_visualizations(self, results: Dict[str, Any]) -> List[str]:
        """Create comprehensive foundry architecture visualizations."""
        
        print("Creating foundry architecture visualizations...")
        generated_files = []
        
        # Figure 1: Processor Density Scaling
        fig_path = self._create_processor_density_scaling(results)
        generated_files.append(fig_path)
        
        # Figure 2: Quantum Coherence Analysis
        fig_path = self._create_quantum_coherence_analysis(results)
        generated_files.append(fig_path)
        
        # Figure 3: Energy Efficiency Analysis
        fig_path = self._create_energy_efficiency_analysis(results)
        generated_files.append(fig_path)
        
        # Figure 4: Industrial Scalability Assessment
        fig_path = self._create_industrial_scalability_assessment(results)
        generated_files.append(fig_path)
        
        return generated_files
    
    def create_virtual_acceleration_visualizations(self, results: Dict[str, Any]) -> List[str]:
        """Create comprehensive virtual acceleration visualizations."""
        
        print("Creating virtual acceleration visualizations...")
        generated_files = []
        
        # Figure 1: Frequency Achievement Analysis
        fig_path = self._create_frequency_achievement_analysis(results)
        generated_files.append(fig_path)
        
        # Figure 2: Temporal Precision Analysis
        fig_path = self._create_temporal_precision_analysis(results)
        generated_files.append(fig_path)
        
        # Figure 3: Parallel Processing Scalability
        fig_path = self._create_parallel_processing_scalability(results)
        generated_files.append(fig_path)
        
        # Figure 4: Virtual Processing Efficiency
        fig_path = self._create_virtual_processing_efficiency(results)
        generated_files.append(fig_path)
        
        return generated_files
    
    def _create_executive_summary_dashboard(self, compression_results, network_results, foundry_results, virtual_results):
        """Create executive summary dashboard."""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Buhera Framework Validation - Executive Summary', fontsize=20, fontweight='bold')
        
        # Subplot 1: Overall Validation Scores
        ax = axes[0, 0]
        components = ['Compression', 'Network\nEvolution']
        scores = [
            compression_results["validation_summary"]["quantitative_results"]["overall_validation_score"],
            network_results["learning_analysis"]["learning_score"]
        ]
        
        if foundry_results:
            components.append('Foundry\nArchitecture')
            scores.append(foundry_results["validation_summary"]["quantitative_results"]["average_validation_score"])
        
        if virtual_results:
            components.append('Virtual\nAcceleration')
            scores.append(virtual_results["validation_summary"]["quantitative_results"]["average_validation_score"])
        
        bars = ax.bar(components, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(components)])
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Validation Threshold')
        ax.set_ylabel('Validation Score')
        ax.set_title('Component Validation Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 2: Breakthrough Validation Status
        ax = axes[0, 1]
        breakthroughs = [
            'Storage =\nUnderstanding',
            'Meta-Information\nCascade',
            'Context\nProcessing',
            'Navigation\nRetrieval'
        ]
        
        validation_status = [
            compression_results["validation_summary"]["key_claims_validated"]["storage_understanding_equivalence"],
            compression_results["validation_summary"]["key_claims_validated"]["superior_compression_through_understanding"],
            compression_results["validation_summary"]["key_claims_validated"]["context_dependent_processing"],
            compression_results["validation_summary"]["key_claims_validated"]["navigation_based_retrieval"]
        ]
        
        if foundry_results:
            breakthroughs.extend(['Molecular\nProcessing', 'Quantum\nCoherence'])
            validation_status.extend([
                foundry_results["validation_summary"]["foundry_claims_validated"]["processor_density_10e9_per_m3"],
                foundry_results["validation_summary"]["foundry_claims_validated"]["room_temperature_quantum_coherence"]
            ])
        
        if virtual_results:
            breakthroughs.extend(['10^30 Hz\nProcessing', 'Femtosecond\nPrecision'])
            validation_status.extend([
                virtual_results["validation_summary"]["acceleration_claims_validated"]["frequency_10e30_hz"],
                virtual_results["validation_summary"]["acceleration_claims_validated"]["femtosecond_precision"]
            ])
        
        colors = ['green' if validated else 'red' for validated in validation_status]
        bars = ax.bar(range(len(breakthroughs)), [1 if v else 0.5 for v in validation_status], color=colors, alpha=0.7)
        ax.set_xticks(range(len(breakthroughs)))
        ax.set_xticklabels(breakthroughs, rotation=45, ha='right')
        ax.set_ylabel('Validation Status')
        ax.set_title('Key Breakthroughs Validation')
        ax.set_ylim(0, 1.2)
        
        # Add status labels
        for i, (bar, validated) in enumerate(zip(bars, validation_status)):
            status_text = '✓ VALIDATED' if validated else '✗ FAILED'
            ax.text(i, bar.get_height() + 0.05, status_text, ha='center', va='bottom', 
                   fontweight='bold', color='white' if validated else 'black')
        
        # Subplot 3: Performance Metrics Radar
        ax = axes[0, 2]
        self._create_radar_chart(ax, compression_results, network_results, foundry_results, virtual_results)
        
        # Subplot 4: Compression Improvement
        ax = axes[1, 0]
        datasets = [dr["name"] for dr in compression_results["compression_comparisons"]["datasets"]]
        improvements = []
        for dataset_result in compression_results["compression_comparisons"]["datasets"]:
            improvements.append(dataset_result["improvement"]["compression_improvement"])
        
        bars = ax.bar(datasets, improvements, color='skyblue', alpha=0.8)
        ax.set_ylabel('Compression Improvement (%)')
        ax.set_title('Compression Improvement by Dataset')
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{improvement:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 5: Understanding Evolution
        ax = axes[1, 1]
        if 'accumulation_results' in network_results:
            sequence_indices = [r["sequence_index"] for r in network_results["accumulation_results"]["accumulation_results"]]
            understanding_scores = [r["understanding_score"] for r in network_results["accumulation_results"]["accumulation_results"]]
            storage_efficiency = [r["storage_efficiency"] for r in network_results["accumulation_results"]["accumulation_results"]]
            
            ax.plot(sequence_indices, understanding_scores, 'o-', linewidth=2, label='Understanding Score')
            ax.plot(sequence_indices, storage_efficiency, 's-', linewidth=2, label='Storage Efficiency')
            ax.set_xlabel('Learning Sequence')
            ax.set_ylabel('Score')
            ax.set_title('Understanding Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Subplot 6: Overall Framework Status
        ax = axes[1, 2]
        total_components = 2
        validated_components = int(compression_results["validation_summary"]["validation_status"]["framework_validated"]) + \
                            int(network_results["learning_analysis"]["validation_success"])
        
        if foundry_results:
            total_components += 1
            validated_components += int(foundry_results["validation_summary"]["validation_status"]["foundry_validated"])
        
        if virtual_results:
            total_components += 1
            validated_components += int(virtual_results["validation_summary"]["validation_status"]["acceleration_validated"])
        
        # Create pie chart
        sizes = [validated_components, total_components - validated_components]
        labels = [f'Validated\n({validated_components})', f'Failed\n({total_components - validated_components})']
        colors = ['lightgreen', 'lightcoral']
        
        if total_components - validated_components == 0:
            sizes = [validated_components]
            labels = [f'All Validated\n({validated_components}/{total_components})']
            colors = ['lightgreen']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Overall Framework Validation Status')
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "executive_summary_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def _create_detailed_performance_dashboard(self, compression_results, network_results, foundry_results, virtual_results):
        """Create detailed performance metrics dashboard."""
        
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle('Buhera Framework Validation - Detailed Performance Analysis', fontsize=20, fontweight='bold')
        
        # Implementation of detailed performance dashboard with 9 subplots
        # This would include detailed metrics for each component
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "detailed_performance_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def _create_component_analysis_dashboard(self, compression_results, network_results, foundry_results, virtual_results):
        """Create component-by-component analysis dashboard."""
        
        # This would create detailed analysis for each component
        output_path = self.figures_dir / "component_analysis_dashboard.png"
        
        # Create placeholder for now
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Component Analysis Dashboard\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_validation_status_dashboard(self, compression_results, network_results, foundry_results, virtual_results):
        """Create validation status tracking dashboard."""
        
        # This would create validation status tracking
        output_path = self.figures_dir / "validation_status_dashboard.png"
        
        # Create placeholder for now
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Validation Status Dashboard\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_radar_chart(self, ax, compression_results, network_results, foundry_results, virtual_results):
        """Create radar chart for performance metrics."""
        
        categories = ['Compression\nEfficiency', 'Understanding\nAccumulation', 'Context\nProcessing']
        values = [
            compression_results["validation_summary"]["quantitative_results"]["overall_validation_score"],
            network_results["learning_analysis"]["learning_score"],
            compression_results["validation_summary"]["quantitative_results"]["context_processing_effectiveness"]
        ]
        
        if foundry_results:
            categories.append('Foundry\nArchitecture')
            values.append(foundry_results["validation_summary"]["quantitative_results"]["average_validation_score"])
        
        if virtual_results:
            categories.append('Virtual\nAcceleration')
            values.append(virtual_results["validation_summary"]["quantitative_results"]["average_validation_score"])
        
        # Close the radar chart
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(2, 3, 3, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', markersize=8)
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_title('Performance Metrics Overview', pad=20)
        ax.grid(True)
    
    def _create_compression_performance_chart(self, compression_data):
        """Create detailed compression performance chart."""
        
        output_path = self.figures_dir / "compression_performance_detailed.png"
        
        # Create placeholder for now
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Compression Performance Chart\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_understanding_compression_analysis(self, compression_data):
        """Create understanding vs compression analysis."""
        
        output_path = self.figures_dir / "understanding_compression_analysis.png"
        
        # Create placeholder for now  
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Understanding-Compression Analysis\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    # Placeholder implementations for other visualization methods
    def _create_context_detection_analysis(self, results):
        output_path = self.figures_dir / "context_detection_analysis.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Context Detection Analysis\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def _create_navigation_rules_analysis(self, results):
        output_path = self.figures_dir / "navigation_rules_analysis.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Navigation Rules Analysis\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def _create_learning_progression_chart(self, results):
        output_path = self.figures_dir / "learning_progression_detailed.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Learning Progression Chart\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def _create_network_growth_dynamics(self, results):
        output_path = self.figures_dir / "network_growth_dynamics.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Network Growth Dynamics\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def _create_understanding_accumulation_heatmap(self, results):
        output_path = self.figures_dir / "understanding_accumulation_heatmap.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Understanding Accumulation Heatmap\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def _create_storage_pattern_evolution(self, results):
        output_path = self.figures_dir / "storage_pattern_evolution.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Storage Pattern Evolution\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def _create_processor_density_scaling(self, results):
        output_path = self.figures_dir / "processor_density_scaling.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Processor Density Scaling\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def _create_quantum_coherence_analysis(self, results):
        output_path = self.figures_dir / "quantum_coherence_analysis.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Quantum Coherence Analysis\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def _create_energy_efficiency_analysis(self, results):
        output_path = self.figures_dir / "energy_efficiency_analysis.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Energy Efficiency Analysis\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def _create_industrial_scalability_assessment(self, results):
        output_path = self.figures_dir / "industrial_scalability_assessment.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Industrial Scalability Assessment\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def _create_frequency_achievement_analysis(self, results):
        output_path = self.figures_dir / "frequency_achievement_analysis.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Frequency Achievement Analysis\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def _create_temporal_precision_analysis(self, results):
        output_path = self.figures_dir / "temporal_precision_analysis.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Temporal Precision Analysis\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def _create_parallel_processing_scalability(self, results):
        output_path = self.figures_dir / "parallel_processing_scalability.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Parallel Processing Scalability\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def _create_virtual_processing_efficiency(self, results):
        output_path = self.figures_dir / "virtual_processing_efficiency.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Virtual Processing Efficiency\n(Implementation in progress)', 
                ha='center', va='center', fontsize=20, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    def generate_all_visualizations(self, validation_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate all comprehensive visualizations."""
        
        print("\n🎨 GENERATING COMPREHENSIVE VISUALIZATIONS...")
        
        all_generated_files = {
            "dashboards": [],
            "compression": [],
            "network": [],
            "foundry": [],
            "virtual": []
        }
        
        # Extract results
        compression_results = validation_results.get("compression_validation")
        network_results = validation_results.get("network_evolution")
        foundry_results = validation_results.get("foundry_validation")
        virtual_results = validation_results.get("virtual_acceleration")
        
        # Generate comprehensive dashboards
        if compression_results and network_results:
            dashboard_files = self.create_comprehensive_dashboard(
                compression_results, network_results, foundry_results, virtual_results
            )
            all_generated_files["dashboards"].extend(dashboard_files)
        
        # Generate component-specific visualizations
        if compression_results:
            compression_files = self.create_compression_visualizations(compression_results)
            all_generated_files["compression"].extend(compression_files)
        
        if network_results:
            network_files = self.create_network_visualizations(network_results)
            all_generated_files["network"].extend(network_files)
        
        if foundry_results:
            foundry_files = self.create_foundry_visualizations(foundry_results)
            all_generated_files["foundry"].extend(foundry_files)
        
        if virtual_results:
            virtual_files = self.create_virtual_acceleration_visualizations(virtual_results)
            all_generated_files["virtual"].extend(virtual_files)
        
        # Summary
        total_files = sum(len(files) for files in all_generated_files.values())
        print(f"✅ Generated {total_files} visualization files:")
        for category, files in all_generated_files.items():
            if files:
                print(f"   • {category.title()}: {len(files)} files")
        
        return all_generated_files
