"""
Extended Gamma-Sprinkled (GS) Model for Crossover Interference

This module implements the three-component mixture model that accounts for
positive interference, negative interference, and no interference simultaneously.
"""

import random
import math
import numpy as np
from .gamma_distributions import GammaDistribution


class Model:
    """
    Model parameters for the three-component gamma mixture.
    
    Parameters:
    -----------
    nu_minus : float
        Shape parameter for negative interference (0 < ν⁻ < 1)
    nu_plus : float  
        Shape parameter for positive interference (ν⁺ > 1)
    p_minus : float
        Proportion of negative interference events (0 ≤ p⁻ ≤ 1)
    p_plus : float
        Proportion of positive interference events (0 ≤ p⁺ ≤ 1)
        
    Note: p_minus + p_plus ≤ 1, p_zero = 1 - p_minus - p_plus
    """
    
    def __init__(self, nu_minus, nu_plus, p_minus, p_plus):
        self.nu_minus = nu_minus
        self.nu_plus = nu_plus  
        self.p_minus = p_minus
        self.p_plus = p_plus
        self.p_zero = 1 - p_minus - p_plus
        
        if self.p_zero < 0:
            raise ValueError("p_minus + p_plus cannot exceed 1")
        
        # Normalized interference parameter
        self.rho = self._calculate_rho()
    
    def _calculate_rho(self):
        """Calculate normalized interference parameter."""
        nu_max_plus = 20  # Maximum positive interference
        return (1 - (1 - self.nu_minus) * self.p_minus + 
                (self.nu_plus - 1) / (nu_max_plus - 1) * self.p_plus)
    
    def __str__(self):
        return (f"Model(ν⁻={self.nu_minus:.2f}, ν⁺={self.nu_plus:.2f}, "
                f"p⁻={self.p_minus:.2f}, p⁺={self.p_plus:.2f}, ρ={self.rho:.2f})")


class MixtureGammaModel:
    """
    Extended gamma-sprinkled model with three interference components.
    
    This model simulates crossover events as a mixture of three processes:
    1. No interference (ν₀ = 1, Haldane model)
    2. Negative interference (0 < ν⁻ < 1, clustering)  
    3. Positive interference (ν⁺ > 1, repulsion)
    """
    
    def __init__(self, chromosome_length=120.0):
        self.chromosome_length = chromosome_length
        self.max_crossovers = 5  # For statistical analysis
    
    def create_gamma_objects(self, nu_minus, nu_plus):
        """
        Create the three gamma distribution objects.
        
        Returns:
        --------
        tuple : (gamma_haldane, gamma_negative, gamma_positive)
        """
        extension = 300  # Extension for proper simulation
        
        # Haldane (no interference)
        gamma_haldane = GammaDistribution(
            shape=1.0, 
            chromosome_length=self.chromosome_length + extension
        )
        
        # Negative interference  
        gamma_negative = GammaDistribution(
            shape=nu_minus,
            chromosome_length=self.chromosome_length + extension
        )
        
        # Positive interference
        gamma_positive = GammaDistribution(
            shape=nu_plus,
            chromosome_length=self.chromosome_length + extension
        )
        
        return gamma_haldane, gamma_negative, gamma_positive
    
    def simulate_crossover_events(self, model, individual_id=-1):
        """
        Simulate crossover events using the three-component mixture model.
        
        Parameters:
        -----------
        model : Model
            Model parameters
        individual_id : int, optional
            Individual identifier for reproducible simulation
            
        Returns:
        --------
        list : Sorted crossover positions
        """
        gamma_haldane, gamma_negative, gamma_positive = self.create_gamma_objects(
            model.nu_minus, model.nu_plus
        )
        
        # Simulate events for each component
        positions_haldane = gamma_haldane.simulate_crossover_positions(
            expected_rate=model.p_zero
        )
        positions_negative = gamma_negative.simulate_crossover_positions(
            expected_rate=model.p_minus  
        )
        positions_positive = gamma_positive.simulate_crossover_positions(
            expected_rate=model.p_plus
        )
        
        # Combine and sort all positions
        all_positions = []
        all_positions.extend(positions_haldane)
        all_positions.extend(positions_negative) 
        all_positions.extend(positions_positive)
        
        # Filter positions within chromosome bounds
        valid_positions = [pos for pos in all_positions 
                          if 0 <= pos <= self.chromosome_length]
        valid_positions.sort()
        
        return valid_positions
    
    def simulate_population(self, model, population_size):
        """
        Simulate crossover data for entire population.
        
        Parameters:
        -----------
        model : Model
            Model parameters
        population_size : int
            Number of individuals to simulate
            
        Returns:
        --------
        list : List of crossover position vectors for each individual
        """
        population_data = []
        
        for i in range(population_size):
            crossover_positions = self.simulate_crossover_events(model)
            population_data.append(crossover_positions)
            
            if i % 1000 == 0 and i > 0:
                print(f"Simulated {i}/{population_size} individuals")
        
        return population_data
    
    def estimate_parameters(self, observed_data, parameter_grid=None):
        """
        Estimate model parameters using maximum likelihood approach.
        
        Parameters:
        -----------
        observed_data : list
            Observed crossover data (list of position vectors)
        parameter_grid : dict, optional
            Parameter search grid
            
        Returns:
        --------
        Model : Best fitting model parameters
        """
        if parameter_grid is None:
            parameter_grid = self._default_parameter_grid()
        
        best_model = None
        best_likelihood = float('-inf')
        
        from .statistical_tests import StatisticalTests
        from .data_processing import PopulationStatistics
        
        stats_calculator = StatisticalTests()
        obs_stats = PopulationStatistics(self.chromosome_length, self.max_crossovers)
        obs_stats.calculate_statistics(observed_data)
        
        print("Searching for optimal parameters...")
        evaluated_models = 0
        
        for nu_minus in parameter_grid['nu_minus']:
            for nu_plus in parameter_grid['nu_plus']:
                for p_minus in parameter_grid['p_minus']:
                    for p_plus in parameter_grid['p_plus']:
                        if p_minus + p_plus <= 1:
                            model = Model(nu_minus, nu_plus, p_minus, p_plus)
                            
                            # Simulate data for this model
                            sim_data = self.simulate_population(model, 5000)
                            sim_stats = PopulationStatistics(
                                self.chromosome_length, self.max_crossovers
                            )
                            sim_stats.calculate_statistics(sim_data)
                            
                            # Calculate likelihood
                            p1, p2 = stats_calculator.compare_distributions(
                                obs_stats.crossover_counts,
                                obs_stats.distance_distribution,
                                sim_stats.crossover_counts,
                                sim_stats.distance_distribution,
                                len(observed_data), 5000
                            )
                            
                            likelihood = min(p1, p2)
                            
                            if likelihood > best_likelihood:
                                best_likelihood = likelihood
                                best_model = model
                            
                            evaluated_models += 1
                            if evaluated_models % 100 == 0:
                                print(f"Evaluated {evaluated_models} models, "
                                      f"best likelihood: {best_likelihood:.4f}")
        
        return best_model
    
    def _default_parameter_grid(self):
        """Default parameter grid for parameter estimation."""
        return {
            'nu_minus': [0.2, 0.4, 0.6, 0.8, 0.95],
            'nu_plus': [1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0],
            'p_minus': [x/10 for x in range(0, 10)],  # 0.0 to 0.9
            'p_plus': [x/10 for x in range(0, 10)]    # 0.0 to 0.9
        }
    
    def test_model_adequacy(self, model, observed_data, alpha=0.05):
        """
        Test if the model adequately fits the observed data.
        
        Parameters:
        -----------
        model : Model
            Model to test
        observed_data : list
            Observed crossover data
        alpha : float
            Significance level
            
        Returns:
        --------
        dict : Test results including p-values
        """
        from .statistical_tests import StatisticalTests
        from .data_processing import PopulationStatistics
        
        # Calculate observed statistics
        obs_stats = PopulationStatistics(self.chromosome_length, self.max_crossovers)
        obs_stats.calculate_statistics(observed_data)
        
        # Simulate data under the model
        sim_data = self.simulate_population(model, len(observed_data))
        sim_stats = PopulationStatistics(self.chromosome_length, self.max_crossovers)
        sim_stats.calculate_statistics(sim_data)
        
        # Perform statistical tests
        stats_calculator = StatisticalTests()
        p1, p2 = stats_calculator.compare_distributions(
            obs_stats.crossover_counts,
            obs_stats.distance_distribution,
            sim_stats.crossover_counts, 
            sim_stats.distance_distribution,
            len(observed_data), len(sim_data)
        )
        
        return {
            'chi2_pvalue': p1,
            'ks_pvalue': p2,
            'model_adequate': min(p1, p2) >= alpha,
            'model': model
        }