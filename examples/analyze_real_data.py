#!/usr/bin/env python3
"""
Example: Analyze Real Crossover Data

This script demonstrates how to analyze real genetic mapping data using the
extended gamma-sprinkled model to detect and quantify crossover interference.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src import MixtureGammaModel, Model, PopulationStatistics, StatisticalTests, DataLoader


def analyze_genetic_map_data(filename, chromosome_length, allele1='H', allele2='B'):
    """
    Analyze crossover data from a genetic mapping file.
    
    Parameters:
    -----------
    filename : str
        Path to genetic map file
    chromosome_length : float
        Chromosome length in cM
    allele1, allele2 : str
        Allele symbols in the data
    """
    
    print(f"Analyzing genetic map data from: {filename}")
    print(f"Chromosome length: {chromosome_length} cM")
    print("=" * 70)
    
    # Load data
    try:
        crossover_data = DataLoader.load_from_genetic_map(
            filename, chromosome_length, allele1, allele2
        )
        print(f"Successfully loaded data for {len(crossover_data)} individuals")
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        print("Creating simulated example data instead...")
        crossover_data = create_example_data(chromosome_length)
    
    # Calculate observed statistics
    observed_stats = PopulationStatistics(chromosome_length)
    observed_stats.calculate_statistics(crossover_data)
    
    summary = observed_stats.get_summary()
    print(f"\nData Summary:")
    print(f"  Number of individuals: {summary['n_individuals']}")
    print(f"  Mean crossovers per individual: {summary['mean_crossovers_observed']:.2f}")
    print(f"  Crossover number distribution: {[f'{p:.3f}' for p in summary['crossover_proportions']]}")
    print(f"  Distance observations: {summary['n_distance_observations']}")
    
    return crossover_data, observed_stats


def test_interference_models(observed_stats, chromosome_length):
    """Test different interference models against observed data."""
    
    print("\n\nTesting Interference Models:")
    print("=" * 70)
    
    # Define models to test
    test_models = {
        'Haldane (No interference)': Model(0.5, 2.0, 0.0, 0.0),
        'Kosambi (Positive interference)': Model(0.5, 2.0, 0.0, 1.0),
        'Negative interference only': Model(0.5, 2.0, 1.0, 0.0),
        'Standard GS model': Model(0.5, 2.0, 0.0, 0.7),
        'Extended GS model': Model(0.6, 2.0, 0.2, 0.4)
    }
    
    mixture_model = MixtureGammaModel(chromosome_length)
    stats_calculator = StatisticalTests()
    
    results = {}
    
    for model_name, model in test_models.items():
        print(f"\nTesting: {model_name}")
        print(f"  Parameters: {model}")
        
        # Simulate data under this model for comparison
        simulated_data = mixture_model.simulate_population(model, observed_stats.n_individuals)
        sim_stats = PopulationStatistics(chromosome_length)
        sim_stats.calculate_statistics(simulated_data)
        
        # Compare with observed data
        p1, p2 = stats_calculator.compare_distributions(
            observed_stats.crossover_counts,
            observed_stats.distance_distribution,
            sim_stats.crossover_counts,
            sim_stats.distance_distribution,
            observed_stats.n_individuals,
            sim_stats.n_individuals
        )
        
        results[model_name] = {
            'model': model,
            'chi2_pvalue': p1,
            'ks_pvalue': p2,
            'adequate': min(p1, p2) >= 0.05,
            'simulated_stats': sim_stats
        }
        
        print(f"  Chi-square p-value: {p1:.6f}")
        print(f"  Kolmogorov-Smirnov p-value: {p2:.6f}")
        print(f"  Model adequate: {results[model_name]['adequate']}")
    
    return results


def parameter_estimation_example(observed_data, chromosome_length):
    """Demonstrate parameter estimation for the extended GS model."""
    
    print("\n\nParameter Estimation:")
    print("=" * 70)
    
    mixture_model = MixtureGammaModel(chromosome_length)
    
    # Use a smaller parameter grid for faster computation
    parameter_grid = {
        'nu_minus': [0.4, 0.6, 0.8],
        'nu_plus': [1.5, 2.0, 3.0, 5.0],
        'p_minus': [0.0, 0.2, 0.4],
        'p_plus': [0.0, 0.3, 0.6]
    }
    
    print("Estimating parameters (this may take a few minutes)...")
    
    try:
        best_model = mixture_model.estimate_parameters(observed_data, parameter_grid)
        
        print(f"\nBest fitting model:")
        print(f"  {best_model}")
        
        # Test the estimated model
        test_result = mixture_model.test_model_adequacy(best_model, observed_data)
        
        print(f"\nModel adequacy test:")
        print(f"  Chi-square p-value: {test_result['chi2_pvalue']:.6f}")
        print(f"  Kolmogorov-Smirnov p-value: {test_result['ks_pvalue']:.6f}")
        print(f"  Model adequate: {test_result['model_adequate']}")
        
        return best_model
        
    except Exception as e:
        print(f"Parameter estimation failed: {e}")
        print("Using default parameters for demonstration...")
        return Model(0.6, 2.0, 0.2, 0.4)


def plot_model_comparison(observed_stats, model_results, chromosome_length):
    """Create plots comparing observed data with different models."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Crossover number distributions
    ax1.set_title('Crossover Number Distributions')
    ax1.set_xlabel('Number of Crossovers')
    ax1.set_ylabel('Proportion')
    
    # Observed data
    x = range(len(observed_stats.crossover_counts))
    ax1.plot(x, observed_stats.crossover_counts, 'ko-', linewidth=2, 
             markersize=8, label='Observed', zorder=10)
    
    # Model predictions
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (model_name, result) in enumerate(model_results.items()):
        if i < len(colors):
            sim_stats = result['simulated_stats']
            ax1.plot(x, sim_stats.crossover_counts, '--', color=colors[i], 
                    alpha=0.7, label=f"{model_name} (p={result['chi2_pvalue']:.3f})")
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Model adequacy p-values
    ax2.set_title('Model Adequacy Test Results')
    ax2.set_ylabel('p-value')
    
    model_names = list(model_results.keys())
    chi2_pvals = [result['chi2_pvalue'] for result in model_results.values()]
    ks_pvals = [result['ks_pvalue'] for result in model_results.values()]
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    ax2.bar(x_pos - width/2, chi2_pvals, width, label='Chi-square test', alpha=0.7)
    ax2.bar(x_pos + width/2, ks_pvals, width, label='Kolmogorov-Smirnov test', alpha=0.7)
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Distance distributions
    ax3.set_title('Inter-crossover Distance Distributions')
    ax3.set_xlabel('Distance (cM)')
    ax3.set_ylabel('Cumulative Probability')
    
    # Show first 50 cM
    x_dist = observed_stats.x_positions[:50]
    y_obs = observed_stats.distance_distribution[:50]
    ax3.plot(x_dist, y_obs, 'k-', linewidth=3, label='Observed', zorder=10)
    
    for i, (model_name, result) in enumerate(model_results.items()):
        if i < len(colors):
            sim_stats = result['simulated_stats']
            y_sim = sim_stats.distance_distribution[:50]
            ax3.plot(x_dist, y_sim, '--', color=colors[i], alpha=0.7, 
                    label=f"{model_name}")
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Interference parameters
    ax4.set_title('Model Interference Parameters')
    ax4.set_ylabel('Parameter Value')
    
    params_data = {
        'ν⁻ (negative)': [],
        'ν⁺ (positive)': [],
        'p⁻ (proportion neg.)': [],
        'p⁺ (proportion pos.)': [],
        'ρ (normalized)': []
    }
    
    for result in model_results.values():
        model = result['model']
        params_data['ν⁻ (negative)'].append(model.nu_minus)
        params_data['ν⁺ (positive)'].append(model.nu_plus)
        params_data['p⁻ (proportion neg.)'].append(model.p_minus)
        params_data['p⁺ (proportion pos.)'].append(model.p_plus)
        params_data['ρ (normalized)'].append(model.rho)
    
    x_pos = np.arange(len(model_names))
    width = 0.15
    
    for i, (param_name, values) in enumerate(params_data.items()):
        ax4.bar(x_pos + i*width - 2*width, values, width, 
               label=param_name, alpha=0.7)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=45)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_example_data(chromosome_length):
    """Create example data for demonstration when real data is not available."""
    
    print("Creating simulated example data...")
    
    # Simulate data with mixed interference
    example_model = Model(0.6, 2.0, 0.2, 0.4)
    mixture_model = MixtureGammaModel(chromosome_length)
    
    return mixture_model.simulate_population(example_model, 500)


def main():
    """Main function to run real data analysis example."""
    
    print("Extended Gamma-Sprinkled Model: Real Data Analysis")
    print("=" * 70)
    
    # Example parameters - modify these for your data
    filename = "data/sample_datasets/example_genetic_map.txt"
    chromosome_length = 200.0
    allele1 = 'H'
    allele2 = 'B'
    
    # Load and analyze data
    crossover_data, observed_stats = analyze_genetic_map_data(
        filename, chromosome_length, allele1, allele2
    )
    
    # Test different models
    model_results = test_interference_models(observed_stats, chromosome_length)
    
    # Parameter estimation
    best_model = parameter_estimation_example(crossover_data, chromosome_length)
    
    # Create comparison plots
    plot_model_comparison(observed_stats, model_results, chromosome_length)
    
    # Summary
    print("\n\nAnalysis Summary:")
    print("=" * 70)
    
    adequate_models = [name for name, result in model_results.items() 
                      if result['adequate']]
    
    if adequate_models:
        print("Models that adequately fit the data:")
        for model_name in adequate_models:
            result = model_results[model_name]
            print(f"  - {model_name}: χ² p={result['chi2_pvalue']:.4f}, "
                  f"KS p={result['ks_pvalue']:.4f}")
    else:
        print("No models provided adequate fit to the data.")
        print("Consider:")
        print("  - Extended parameter search")
        print("  - Data quality assessment")
        print("  - Alternative models")
    
    print(f"\nBest estimated model: {best_model}")
    print("\nGenerated files:")
    print("  - model_comparison.png")


if __name__ == "__main__":
    main()