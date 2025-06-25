#!/usr/bin/env python3
"""
Example: Simulate Crossover Data using Extended GS Model

This script demonstrates how to simulate crossover data using the three-component
gamma-sprinkled model and compare different interference scenarios.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src import MixtureGammaModel, Model, PopulationStatistics, StatisticalTests


def simulate_different_scenarios():
    """Simulate data under different interference scenarios."""
    
    # Set up scenarios
    scenarios = {
        'Haldane (No Interference)': Model(0.5, 2.0, 0.0, 0.0),
        'Positive Interference Only': Model(0.5, 2.0, 0.0, 1.0),
        'Negative Interference Only': Model(0.5, 2.0, 1.0, 0.0),
        'Mixed Interference': Model(0.6, 2.0, 0.2, 0.4)
    }
    
    chromosome_length = 200.0
    population_size = 2000
    
    # Initialize model
    mixture_model = MixtureGammaModel(chromosome_length)
    
    print("Simulating crossover data under different interference scenarios...")
    print("=" * 70)
    
    results = {}
    
    for scenario_name, model in scenarios.items():
        print(f"\nScenario: {scenario_name}")
        print(f"Model parameters: {model}")
        
        # Simulate data
        crossover_data = mixture_model.simulate_population(model, population_size)
        
        # Calculate statistics
        stats = PopulationStatistics(chromosome_length)
        stats.calculate_statistics(crossover_data)
        
        # Store results
        results[scenario_name] = {
            'model': model,
            'data': crossover_data,
            'stats': stats
        }
        
        # Print summary
        summary = stats.get_summary()
        print(f"  Population size: {summary['n_individuals']}")
        print(f"  Mean crossovers: {summary['mean_crossovers_observed']:.2f}")
        print(f"  Crossover distribution: {[f'{p:.3f}' for p in summary['crossover_proportions']]}")
    
    return results


def compare_models(results):
    """Compare different models using statistical tests."""
    
    print("\n\nModel Comparison Results:")
    print("=" * 70)
    
    stats_calculator = StatisticalTests()
    
    # Use Haldane as reference
    reference_name = 'Haldane (No Interference)'
    reference_stats = results[reference_name]['stats']
    
    for scenario_name, scenario_results in results.items():
        if scenario_name == reference_name:
            continue
        
        test_stats = scenario_results['stats']
        
        # Compare distributions
        p1, p2 = stats_calculator.compare_distributions(
            reference_stats.crossover_counts,
            reference_stats.distance_distribution,
            test_stats.crossover_counts,
            test_stats.distance_distribution,
            reference_stats.n_individuals,
            test_stats.n_individuals
        )
        
        print(f"\n{scenario_name} vs {reference_name}:")
        print(f"  Chi-square test p-value: {p1:.6f}")
        print(f"  Kolmogorov-Smirnov test p-value: {p2:.6f}")
        print(f"  Models significantly different: {min(p1, p2) < 0.05}")


def plot_results(results):
    """Create plots comparing different scenarios."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Crossover number distributions
    ax1.set_title('Crossover Number Distributions')
    ax1.set_xlabel('Number of Crossovers')
    ax1.set_ylabel('Proportion')
    
    for scenario_name, scenario_results in results.items():
        counts = scenario_results['stats'].crossover_counts
        x = range(len(counts))
        ax1.plot(x, counts, marker='o', label=scenario_name)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean crossovers comparison
    ax2.set_title('Mean Crossovers per Individual')
    ax2.set_ylabel('Mean Crossovers')
    
    scenario_names = []
    mean_crossovers = []
    
    for scenario_name, scenario_results in results.items():
        scenario_names.append(scenario_name.replace(' ', '\n'))
        mean_crossovers.append(scenario_results['stats'].mean_crossovers_observed)
    
    bars = ax2.bar(range(len(scenario_names)), mean_crossovers)
    ax2.set_xticks(range(len(scenario_names)))
    ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_crossovers):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    # Plot 3: Distance distributions (CDF)
    ax3.set_title('Inter-crossover Distance Distributions')
    ax3.set_xlabel('Distance (cM)')
    ax3.set_ylabel('Cumulative Probability')
    
    for scenario_name, scenario_results in results.items():
        stats = scenario_results['stats']
        x = stats.x_positions[:50]  # Show first 50 cM
        y = stats.distance_distribution[:50]
        ax3.plot(x, y, label=scenario_name)
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Interference parameter comparison
    ax4.set_title('Normalized Interference Parameter (ρ)')
    ax4.set_ylabel('ρ value')
    
    rho_values = []
    for scenario_name, scenario_results in results.items():
        rho_values.append(scenario_results['model'].rho)
    
    bars = ax4.bar(range(len(scenario_names)), rho_values)
    ax4.set_xticks(range(len(scenario_names)))
    ax4.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No interference')
    ax4.legend()
    
    # Add value labels
    for bar, value in zip(bars, rho_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('simulation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def power_analysis_example():
    """Demonstrate power analysis for detecting interference."""
    
    print("\n\nPower Analysis Example:")
    print("=" * 70)
    
    # Define true and null models
    true_model = Model(0.6, 2.0, 0.3, 0.4)  # Mixed interference
    null_model = Model(0.5, 2.0, 0.0, 0.0)  # No interference (Haldane)
    
    chromosome_length = 200.0
    sample_sizes = [50, 100, 200, 500, 1000]
    
    print(f"True model: {true_model}")
    print(f"Null model: {null_model}")
    print(f"Chromosome length: {chromosome_length} cM")
    
    # Calculate power
    stats_calculator = StatisticalTests()
    power_results = stats_calculator.power_analysis(
        true_model, null_model, chromosome_length, sample_sizes, 
        n_simulations=50, alpha=0.05
    )
    
    print("\nPower Analysis Results:")
    print("Sample Size\tPower")
    print("-" * 20)
    for n, power in power_results.items():
        print(f"{n}\t\t{power:.3f}")
    
    # Plot power curve
    plt.figure(figsize=(8, 6))
    plt.plot(list(power_results.keys()), list(power_results.values()), 
             marker='o', linewidth=2, markersize=8)
    plt.xlabel('Sample Size')
    plt.ylabel('Statistical Power')
    plt.title('Power to Detect Mixed Interference vs No Interference')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Power')
    plt.legend()
    plt.savefig('power_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run all examples."""
    
    print("Extended Gamma-Sprinkled Model: Simulation Examples")
    print("=" * 70)
    
    # Simulate different scenarios
    results = simulate_different_scenarios()
    
    # Compare models statistically
    compare_models(results)
    
    # Create plots
    plot_results(results)
    
    # Power analysis
    power_analysis_example()
    
    print("\n\nSimulation complete!")
    print("Generated files:")
    print("  - simulation_comparison.png")
    print("  - power_analysis.png")


if __name__ == "__main__":
    main()