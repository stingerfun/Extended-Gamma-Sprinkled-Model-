"""
Gamma Distribution Module for Crossover Interference Analysis

This module implements gamma distribution-based simulation of crossover events
along chromosomes, accounting for different levels of interference.
"""

import numpy as np
import random
import math


class GammaDistribution:
    """
    Implements gamma distribution for modeling crossover interference.
    
    Parameters:
    -----------
    shape : float
        Shape parameter (ν) of gamma distribution
        - ν > 1: positive interference
        - ν = 1: no interference (Haldane model)  
        - 0 < ν < 1: negative interference
    chromosome_length : float
        Length of chromosome in centiMorgans (cM)
    """
    
    def __init__(self, shape=1.0, chromosome_length=120.0):
        self.shape = shape  # ν parameter
        self.scale = 100.0 / shape  # θ parameter  
        self.chromosome_length = chromosome_length
        
        # Precalculated distribution tables
        self.positions = []
        self.density = []
        self.cumulative = []
        
        self._build_distribution_table()
    
    def _build_distribution_table(self, step=0.01, extension=300):
        """Build lookup tables for gamma distribution."""
        max_distance = self.chromosome_length + extension
        
        self.positions = []
        self.density = []
        self.cumulative = []
        
        # Handle special case for ν < 1 at x=0
        x = 0
        if self.shape >= 1:
            f = self._gamma_density(0)
        else:
            f = 0  # Avoid infinity for ν < 1
        
        self.positions.append(x)
        self.density.append(f)
        self.cumulative.append(0)
        
        # Build table with small first step
        x = step
        f = self._gamma_density(x)
        cum = self._small_x_cumulative(x)
        
        self.positions.append(x)
        self.density.append(f)
        self.cumulative.append(cum)
        
        # Continue building table
        f_prev = f
        while x <= max_distance:
            x += step
            f = self._gamma_density(x)
            
            # Trapezoidal integration
            delta_cum = step * (f + f_prev) * 0.5
            cum += delta_cum
            
            self.positions.append(x)
            self.density.append(f)
            self.cumulative.append(cum)
            
            f_prev = f
    
    def _gamma_density(self, x):
        """Calculate gamma distribution density at point x."""
        if x <= 0:
            return 0 if self.shape >= 1 else float('inf')
        
        gamma_k = math.gamma(self.shape)
        scale_pow_k = math.pow(self.scale, self.shape)
        x_pow_km1 = math.pow(x, self.shape - 1)
        exp_term = math.exp(-x / self.scale)
        
        return (1.0 / (gamma_k * scale_pow_k)) * x_pow_km1 * exp_term
    
    def _small_x_cumulative(self, x):
        """Calculate cumulative distribution for small x using asymptotic expansion."""
        gamma_k = math.gamma(self.shape)
        scale_pow_k = math.pow(self.scale, self.shape)
        x_pow_k = math.pow(x, self.shape)
        x_pow_kp1 = math.pow(x, self.shape + 1)
        x_pow_kp2 = math.pow(x, self.shape + 2)
        
        factor = 1.0 / (gamma_k * scale_pow_k)
        term1 = x_pow_k / self.shape
        term2 = x_pow_kp1 / (self.scale * (self.shape + 1))
        term3 = x_pow_kp2 / (self.scale * self.scale * (self.shape + 2) * 2)
        
        return factor * (term1 - term2 + term3)
    
    def generate_random_distance(self):
        """Generate random distance following gamma distribution."""
        if self.shape == 1:
            # Exponential distribution (Haldane model)
            u = random.uniform(0, 1)
            if u > 0.9999999:
                u = 0.9999999
            return -self.scale * math.log(1 - u)
        
        # Use lookup table for general case
        u = random.uniform(0, 1)
        return self._inverse_cumulative(u)
    
    def _inverse_cumulative(self, u):
        """Find x such that F(x) = u using lookup table."""
        n = len(self.cumulative)
        
        if u <= self.cumulative[0]:
            return self.positions[0]
        if u >= self.cumulative[n-1]:
            return self.positions[n-1]
        
        # Binary search
        i_min, i_max = 0, n - 1
        while i_max - i_min > 1:
            i = (i_min + i_max) // 2
            if u >= self.cumulative[i]:
                i_min = i
            else:
                i_max = i
        
        # Linear interpolation
        if i_min == i_max or self.cumulative[i_min] == self.cumulative[i_max]:
            return self.positions[i_min]
        
        weight = (u - self.cumulative[i_min]) / (self.cumulative[i_max] - self.cumulative[i_min])
        return self.positions[i_min] + weight * (self.positions[i_max] - self.positions[i_min])
    
    def simulate_crossover_positions(self, expected_rate=1.0, start_position=None):
        """
        Simulate crossover positions along chromosome.
        
        Parameters:
        -----------
        expected_rate : float
            Expected number of crossovers per 100 cM
        start_position : float, optional
            Starting position offset
            
        Returns:
        --------
        list : Crossover positions in cM
        """
        positions = []
        
        # First crossover position
        if start_position is None:
            # Use random position within first interval
            first_distance = self.generate_random_distance()
            x = random.uniform(0, first_distance)
        else:
            x = start_position
        
        x = x / expected_rate
        
        if 0 <= x <= self.chromosome_length:
            positions.append(x)
        
        # Subsequent crossovers
        while x <= self.chromosome_length:
            distance = self.generate_random_distance() / expected_rate
            x += distance
            
            if 0 <= x <= self.chromosome_length:
                positions.append(x)
        
        return positions
    
    def get_mapping_function_values(self, positions, chromosome_length):
        """
        Calculate mapping function values for given positions.
        
        Parameters:
        -----------
        positions : list
            List of crossover position vectors for multiple individuals
        chromosome_length : float
            Chromosome length
            
        Returns:
        --------
        list : Mapping function values (recombination probabilities)
        """
        x_values = list(range(int(chromosome_length) + 1))
        mapping_values = []
        
        next_crossover_indices = [0] * len(positions)
        
        for x in x_values:
            if x == 0:
                mapping_values.append(0)
                continue
            
            recombinant_count = 0
            for i, individual_positions in enumerate(positions):
                # Count crossovers up to position x
                while (next_crossover_indices[i] < len(individual_positions) and 
                       individual_positions[next_crossover_indices[i]] < x):
                    next_crossover_indices[i] += 1
                
                # Check if odd number of crossovers (recombinant)
                if next_crossover_indices[i] % 2 == 1:
                    recombinant_count += 1
            
            mapping_values.append(recombinant_count / len(positions))
        
        return mapping_values