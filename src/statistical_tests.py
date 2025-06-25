"""
Statistical Tests for Model Comparison

This module implements chi-square and Kolmogorov-Smirnov tests for comparing
observed crossover data with simulated data from different interference models.
"""

import math
import numpy as np
from scipy.stats import chi2


class StatisticalTests:
    """
    Statistical tests for comparing crossover interference models.
    
    Implements chi-square test for crossover number distributions and
    Kolmogorov-Smirnov test for inter-crossover distance distributions.
    """
    
    def __init__(self):
        pass
    
    def chi2_test(self, observed_counts, expected_counts, n_observed, n_expected=None):
        """
        Perform chi-square test for discrete distributions.
        
        Parameters:
        -----------
        observed_counts : list
            Observed proportions of crossover numbers
        expected_counts : list  
            Expected proportions under model
        n_observed : int
            Sample size for observed data
        n_expected : int, optional
            Sample size for expected data (default: n_observed)
            
        Returns:
        --------
        tuple : (p_value, chi2_statistic, degrees_of_freedom)
        """
        if n_expected is None:
            n_expected = n_observed
        
        # Effective sample size for comparison
        n_effective = (n_observed * n_expected) / (n_observed + n_expected)
        
        # Combine categories with low expected counts
        i_min, i_max, p_min_obs, p_min_exp, p_max_obs, p_max_exp = \
            self._combine_low_frequency_categories(
                observed_counts, expected_counts, n_effective
            )
        
        if i_max <= i_min:
            return 1.0, 0.0, 0  # Non-significant if no categories to test
        
        # Calculate chi-square statistic
        chi2_stat = 0.0
        df = i_max - i_min
        
        for i in range(i_min, i_max + 1):
            p_obs = observed_counts[i]
            p_exp = expected_counts[i]
            
            # Use combined categories at boundaries
            if i == i_min:
                p_obs = p_min_obs
                p_exp = p_min_exp
            if i == i_max:
                p_obs = p_max_obs
                p_exp = p_max_exp
            
            if p_exp > 0:
                chi2_stat += n_effective * ((p_obs - p_exp) ** 2) / p_exp
        
        # Calculate p-value
        p_value = 1 - chi2.cdf(chi2_stat, df)
        
        return p_value, chi2_stat, df
    
    def _combine_low_frequency_categories(self, observed, expected, n_effective, min_count=6):
        """Combine categories with expected count < min_count."""
        n_categories = len(observed)
        
        # Find minimum category boundary
        i_min = 0
        p_min_obs = observed[i_min]
        p_min_exp = expected[i_min]
        
        while p_min_exp * n_effective < min_count and i_min < n_categories - 1:
            i_min += 1
            p_min_obs += observed[i_min]
            p_min_exp += expected[i_min]
        
        # Find maximum category boundary  
        i_max = n_categories - 1
        p_max_obs = observed[i_max]
        p_max_exp = expected[i_max]
        
        while p_max_exp * n_effective < min_count and i_max > i_min:
            i_max -= 1
            p_max_obs += observed[i_max]
            p_max_exp += expected[i_max]
        
        return i_min, i_max, p_min_obs, p_min_exp, p_max_obs, p_max_exp
    
    def kolmogorov_smirnov_test(self, observed_cdf, expected_cdf, n_observed, n_expected=None):
        """
        Perform Kolmogorov-Smirnov test for continuous distributions.
        
        Parameters:
        -----------
        observed_cdf : list
            Observed cumulative distribution function values
        expected_cdf : list
            Expected CDF values under model
        n_observed : int
            Sample size for observed data
        n_expected : int, optional
            Sample size for expected data
            
        Returns:
        --------
        tuple : (p_value, max_difference, argmax_index)
        """
        if n_expected is None:
            n_effective = n_observed
        else:
            n_effective = (n_observed * n_expected) / (n_observed + n_expected)
        
        # Calculate maximum difference between CDFs
        max_diff = 0.0
        argmax_idx = 0
        
        for i, (obs, exp) in enumerate(zip(observed_cdf, expected_cdf)):
            diff = abs(obs - exp)
            if diff > max_diff:
                max_diff = diff
                argmax_idx = i
        
        # Calculate test statistic and p-value
        test_statistic = max_diff * math.sqrt(n_effective)
        p_value = self._kolmogorov_distribution_cdf(test_statistic)
        
        return p_value, max_diff, argmax_idx
    
    def _kolmogorov_distribution_cdf(self, x):
        """
        Calculate p-value using Kolmogorov distribution.
        
        Uses standard critical values for common significance levels.
        """
        if x >= 1.94947:
            return 0.001
        elif x >= 1.62762:
            return 0.01
        elif x >= 1.51743:
            return 0.02
        elif x >= 1.35810:
            return 0.05
        elif x >= 1.22385:
            return 0.1
        else:
            return 1.0
    
    def compare_distributions(self, obs_counts, obs_distances, exp_counts, exp_distances,
                            n_obs, n_exp):
        """
        Compare observed and expected distributions using both tests.
        
        Parameters:
        -----------
        obs_counts : list
            Observed crossover number proportions
        obs_distances : list
            Observed distance distribution (CDF)
        exp_counts : list
            Expected crossover number proportions  
        exp_distances : list
            Expected distance distribution (CDF)
        n_obs : int
            Observed sample size
        n_exp : int
            Expected sample size
            
        Returns:
        --------
        tuple : (chi2_pvalue, ks_pvalue)
        """
        # Chi-square test for crossover numbers
        p1, _, _ = self.chi2_test(obs_counts, exp_counts, n_obs, n_exp)
        
        # Kolmogorov-Smirnov test for distances
        p2, _, _ = self.kolmogorov_smirnov_test(obs_distances, exp_distances, n_obs, n_exp)
        
        return p1, p2
    
    def likelihood_ratio_test(self, likelihood_restricted, likelihood_full, 
                            df_restricted, df_full):
        """
        Perform likelihood ratio test between nested models.
        
        Parameters:
        -----------
        likelihood_restricted : float
            Log-likelihood of restricted model
        likelihood_full : float
            Log-likelihood of full model
        df_restricted : int
            Degrees of freedom for restricted model
        df_full : int
            Degrees of freedom for full model
            
        Returns:
        --------
        float : p-value of likelihood ratio test
        """
        lr_statistic = 2 * (likelihood_full - likelihood_restricted)
        df_diff = df_full - df_restricted
        
        if df_diff <= 0:
            return 1.0
        
        p_value = 1 - chi2.cdf(lr_statistic, df_diff)
        return p_value
    
    def power_analysis(self, true_model, test_model, chromosome_length, 
                      sample_sizes, n_simulations=100, alpha=0.05):
        """
        Estimate statistical power for detecting difference between models.
        
        Parameters:
        -----------
        true_model : Model
            True underlying model
        test_model : Model  
            Model being tested against
        chromosome_length : float
            Chromosome length
        sample_sizes : list
            Sample sizes to test
        n_simulations : int
            Number of simulation runs
        alpha : float
            Significance level
            
        Returns:
        --------
        dict : Power estimates for each sample size
        """
        from .mixture_model import MixtureGammaModel
        from .data_processing import PopulationStatistics
        
        power_results = {}
        mixture_model = MixtureGammaModel(chromosome_length)
        
        for n in sample_sizes:
            rejections = 0
            
            for _ in range(n_simulations):
                # Simulate data under true model
                true_data = mixture_model.simulate_population(true_model, n)
                
                # Calculate statistics
                true_stats = PopulationStatistics(chromosome_length, 5)
                true_stats.calculate_statistics(true_data)
                
                # Simulate data under test model
                test_data = mixture_model.simulate_population(test_model, n)
                test_stats = PopulationStatistics(chromosome_length, 5)
                test_stats.calculate_statistics(test_data)
                
                # Compare distributions
                p1, p2 = self.compare_distributions(
                    true_stats.crossover_counts,
                    true_stats.distance_distribution,
                    test_stats.crossover_counts,
                    test_stats.distance_distribution,
                    n, n
                )
                
                # Reject if either test is significant
                if min(p1, p2) < alpha:
                    rejections += 1
            
            power_results[n] = rejections / n_simulations
        
        return power_results