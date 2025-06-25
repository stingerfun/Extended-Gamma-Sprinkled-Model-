# Extended Gamma-Sprinkled Model for Crossover Interference Analysis

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package implements the **extended gamma-sprinkled (GS) model** for analyzing crossover interference in genetic recombination data. Unlike traditional models that only account for positive interference and no interference, this model simultaneously considers three types of crossover interference:

- **Positive interference** (ν > 1): Crossover repulsion
- **No interference** (ν = 1): Independent crossovers (Haldane model)  
- **Negative interference** (0 < ν < 1): Crossover clustering

## 📖 Background

Crossover interference (COI) refers to the non-random distribution of crossover events along chromosomes during meiosis. The standard gamma-sprinkled model accounts for positive interference and no interference but ignores potential negative interference, which has been reported in various organisms including *Neurospora*, yeast, *Drosophila*, *Arabidopsis*, and several crop species.

Our extended model uses a mixture of three gamma distributions to capture the full spectrum of interference patterns that may occur simultaneously on the same chromosome.

## 📑 Citation

If you use this software in your research, please cite:

```
Sapielkin, S., Frenkel, Z., Privman, E., & Korol, A.B. (2025). 
Statistical analysis and simulations that account for simultaneous effects 
of positive, negative, and no crossover interference in multilocus 
recombination data.
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/crossover-interference-analysis.git
cd crossover-interference-analysis

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from src import MixtureGammaModel, Model

# Define a model with mixed interference
# ν⁻=0.6 (negative interference), ν⁺=2.0 (positive interference)
# p⁻=0.2 (20% negative), p⁺=0.4 (40% positive), p₀=0.4 (40% no interference)
model = Model(nu_minus=0.6, nu_plus=2.0, p_minus=0.2, p_plus=0.4)

# Initialize the mixture model
mixture_model = MixtureGammaModel(chromosome_length=200.0)

# Simulate crossover data
crossover_data = mixture_model.simulate_population(model, population_size=1000)

# Test model adequacy
result = mixture_model.test_model_adequacy(model, crossover_data)
print(f"Model adequate: {result['model_adequate']}")
print(f"Chi-square p-value: {result['chi2_pvalue']:.6f}")
print(f"Kolmogorov-Smirnov p-value: {result['ks_pvalue']:.6f}")
```

## 📊 Examples

### Simulate Different Interference Scenarios

```bash
python examples/simulate_data.py
```

This script demonstrates:
- Simulation under different interference models
- Statistical comparison between models
- Power analysis for detecting interference
- Visualization of results

### Analyze Real Genetic Data

```bash
python examples/analyze_real_data.py
```

This script shows how to:
- Load genetic mapping data
- Test different interference models
- Estimate model parameters
- Compare model adequacy

## 🏗️ Package Structure

```
crossover-interference-analysis/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── gamma_distributions.py      # Gamma distribution simulation
│   ├── mixture_model.py            # Extended GS model implementation  
│   ├── statistical_tests.py        # Chi-square and K-S tests
│   └── data_processing.py          # Data processing utilities
├── examples/
│   ├── simulate_data.py            # Simulation examples
│   └── analyze_real_data.py        # Real data analysis
├── data/
│   └── sample_datasets/            # Example datasets
└── requirements.txt                # Dependencies
```

## 🔬 Key Features

### Model Components

1. **GammaDistribution**: Simulates crossover events using gamma distributions with different shape parameters
2. **MixtureGammaModel**: Implements the three-component mixture model
3. **StatisticalTests**: Chi-square and Kolmogorov-Smirnov tests for model comparison
4. **PopulationStatistics**: Calculates statistics from crossover position data

### Interference Parameters

- **ν⁻** (nu_minus): Shape parameter for negative interference (0 < ν⁻ < 1)
- **ν⁺** (nu_plus): Shape parameter for positive interference (ν⁺ > 1)  
- **p⁻** (p_minus): Proportion of negative interference events
- **p⁺** (p_plus): Proportion of positive interference events
- **p₀** (p_zero): Proportion of no-interference events (p₀ = 1 - p⁻ - p⁺)

### Statistical Tests

- **Chi-square test**: Compares crossover number distributions
- **Kolmogorov-Smirnov test**: Compares inter-crossover distance distributions
- **Likelihood ratio test**: For nested model comparisons
- **Power analysis**: Estimates detection power for different sample sizes

## 📈 Model Selection

The package provides several approaches for model selection:

1. **Parameter estimation**: Grid search with maximum likelihood
2. **Model adequacy testing**: Statistical tests against observed data
3. **Information criteria**: For comparing non-nested models
4. **Cross-validation**: For assessing model generalizability

## 📄 Data Formats

### Genetic Map Format

```
# position marker_name genotype1 genotype2 ...
0.0     Marker1      H B H B B
12.5    Marker2      H H B B H  
25.0    Marker3      B H H H B
...
```

### Crossover Positions Format

```
# One line per individual, crossover positions separated by spaces
12.5 67.3 145.2
8.1 89.7
156.3
45.2 78.9 123.4
...
```

## 🎯 Applications

This software is useful for:

- **Genetic mapping studies**: Characterizing recombination patterns
- **Breeding programs**: Understanding crossover distributions
- **Evolutionary genetics**: Studying recombination evolution
- **Genomics research**: Analyzing high-density marker data
- **Comparative genetics**: Cross-species interference comparison

## 📋 Requirements

- Python 3.6+
- NumPy ≥ 1.19.0
- SciPy ≥ 1.5.0  
- Matplotlib ≥ 3.3.0

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/crossover-interference-analysis.git
cd crossover-interference-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt
```

### Running Tests

```bash
python -m pytest tests/
```

## 📚 Theory

### Gamma Distribution Model

The model uses gamma distributions to describe the distances between consecutive crossover events:

```
f(x; ν, θ) = (1/[Γ(ν)θ^ν]) * x^(ν-1) * e^(-x/θ)
```

where:
- ν: shape parameter (interference strength)
- θ: scale parameter (θ = 100/ν for 1 cM = 1% recombination)

### Mixture Model

The extended GS model combines three components:

```
P(x) = p₀ * f(x; 1, θ₀) + p⁻ * f(x; ν⁻, θ⁻) + p⁺ * f(x; ν⁺, θ⁺)
```

### Normalized Interference Parameter

For model comparison, we define a normalized parameter:

```
ρ = 1 - (1 - ν⁻) * p⁻ + [(ν⁺ - 1)/(ν⁺ₘₐₓ - 1)] * p⁺
```

where ρ = 1 indicates no overall interference.

## 🔧 Troubleshooting

### Common Issues

1. **Convergence problems**: Try different parameter grids or increase population size
2. **Memory issues**: Reduce population size or chromosome length for large simulations
3. **Poor model fit**: Check data quality or consider alternative models

### Performance Tips

- Use pre-computed gamma tables for repeated simulations
- Implement parallel processing for parameter estimation
- Consider approximate methods for very large datasets

## 📞 Support

For questions, issues, or collaborations:

- **Email**: ssapielkin@campus.haifa.ac.il
- **Issues**: Open a GitHub issue
- **Documentation**: See examples/ directory

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- University of Haifa, Institute of Evolution
- The research was supported by [funding sources]
- Thanks to the genetic mapping community for valuable feedback

---

**Keywords**: genetic recombination, crossover interference, gamma distribution, statistical genetics, linkage analysis, meiosis