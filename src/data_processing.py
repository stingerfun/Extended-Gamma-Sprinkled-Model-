"""
Data Processing for Crossover Analysis

This module processes crossover position data and calculates statistics
needed for model fitting and comparison.
"""

import numpy as np


class PopulationStatistics:
    """
    Calculate and store statistics from crossover data.
    
    Processes crossover position data to extract:
    - Distribution of crossover numbers per individual
    - Distribution of distances between consecutive crossovers
    - Cumulative distribution functions
    """
    
    def __init__(self, chromosome_length, max_crossovers=5):
        self.chromosome_length = chromosome_length
        self.max_crossovers = max_crossovers
        
        # Position grid for distance analysis
        self.x_positions = list(range(int(chromosome_length) + 2))
        
        # Results
        self.n_individuals = 0
        self.crossover_counts = []  # Proportions with 0,1,2,... crossovers
        self.distance_distribution = []  # CDF of inter-crossover distances
        self.mean_crossovers_observed = 0
        self.mean_crossovers_expected = 0
        
        # Distance data for individuals with exactly 2 crossovers
        self.distances_two_crossovers = []
        
        # Raw distance data by number of crossovers
        self.distances_by_count = []
    
    def calculate_statistics(self, crossover_data, internal_only=True, start_position=0):
        """
        Calculate population statistics from crossover data.
        
        Parameters:
        -----------
        crossover_data : list
            List of crossover position vectors for each individual
        internal_only : bool
            If True, only consider distances between crossovers (not to chromosome ends)
        start_position : float
            Starting position for analysis
        """
        self.n_individuals = len(crossover_data)
        
        # Initialize counters
        crossover_counts = [0] * (self.max_crossovers + 1)
        all_distances = []
        distances_two_co = []
        distances_by_count = [[] for _ in range(self.max_crossovers + 1)]
        
        total_crossovers = 0
        
        for individual_positions in crossover_data:
            # Filter positions within analysis region
            positions = [pos for pos in individual_positions 
                        if start_position <= pos <= self.chromosome_length]
            
            # Count crossovers
            n_crossovers = len(positions)
            total_crossovers += n_crossovers
            
            # Update crossover count distribution
            count_category = min(n_crossovers, self.max_crossovers)
            crossover_counts[count_category] += 1
            
            # Calculate inter-crossover distances
            distances = []
            if len(positions) >= 2:
                prev_pos = start_position if not internal_only else positions[0]
                
                for i, pos in enumerate(positions):
                    if i > 0 or not internal_only:
                        distance = pos - prev_pos
                        distances.append(distance)
                        all_distances.append(distance)
                    prev_pos = pos
                
                # Add final distance to chromosome end if not internal_only
                if not internal_only:
                    final_distance = self.chromosome_length - prev_pos
                    distances.append(final_distance)
                    all_distances.append(final_distance)
            
            # Store distances by crossover count
            if count_category < len(distances_by_count):
                distances_by_count[count_category].extend(distances)
            
            # Special case: exactly 2 crossovers
            if n_crossovers == 2 and len(distances) > 0:
                distances_two_co.append(distances[0])
        
        # Convert counts to proportions
        self.crossover_counts = [count / self.n_individuals for count in crossover_counts]
        
        # Calculate mean crossovers
        self.mean_crossovers_observed = total_crossovers / self.n_individuals
        self.mean_crossovers_expected = sum(i * prop for i, prop in enumerate(self.crossover_counts))
        
        # Calculate distance distributions
        self.distance_distribution = self._calculate_cdf(all_distances)
        self.distances_two_crossovers = distances_two_co
        self.distances_by_count = distances_by_count
    
    def _calculate_cdf(self, distances):
        """Calculate cumulative distribution function for distances."""
        if not distances:
            # Return uniform distribution if no data
            return [float(i + 1) / len(self.x_positions) for i in range(len(self.x_positions))]
        
        distances.sort()
        cdf = []
        n_distances = len(distances)
        distance_idx = 0
        
        for x in self.x_positions:
            # Count distances less than x
            count = 0
            while distance_idx < n_distances and distances[distance_idx] < x:
                count += 1
                distance_idx += 1
            
            # Add to previous count
            if cdf:
                count += cdf[-1] * n_distances
            
            cdf.append(count / n_distances)
        
        return cdf
    
    def get_summary(self):
        """Get summary statistics as dictionary."""
        return {
            'n_individuals': self.n_individuals,
            'mean_crossovers_observed': self.mean_crossovers_observed,
            'mean_crossovers_expected': self.mean_crossovers_expected,
            'crossover_proportions': self.crossover_counts,
            'n_distance_observations': len(self.distances_two_crossovers),
            'chromosome_length': self.chromosome_length
        }
    
    def format_for_output(self, include_headers=False):
        """Format statistics for file output."""
        if include_headers:
            # Headers for crossover counts
            header = ""
            for n in range(self.max_crossovers):
                header += f"\t{n}"
            header += f"\t>={self.max_crossovers}"
            
            # Headers for distance CDF
            for x in self.x_positions:
                header += f"\t{x}"
            
            return header
        
        # Data row
        data = ""
        for prop in self.crossover_counts:
            data += f"\t{prop:.6f}"
        
        for cdf_val in self.distance_distribution:
            data += f"\t{cdf_val:.6f}"
        
        return data


class DataLoader:
    """
    Load crossover data from various file formats.
    """
    
    @staticmethod
    def load_from_genetic_map(filename, chromosome_length, allele1='H', allele2='B'):
        """
        Load crossover data from genetic mapping file.
        
        Expected format: position marker_name genotype1 genotype2 ...
        
        Parameters:
        -----------
        filename : str
            Path to genetic map file
        chromosome_length : float
            Length of chromosome in cM
        allele1, allele2 : str
            Allele symbols in the data
            
        Returns:
        --------
        list : Crossover position data for each individual
        """
        # Read marker data
        marker_positions = []
        marker_names = []
        marker_genotypes = []
        
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                
                try:
                    position = float(parts[0])
                    marker_name = parts[1]
                    genotypes = parts[2:]
                    
                    marker_positions.append(position)
                    marker_names.append(marker_name)
                    
                    # Convert genotypes to 0/1
                    binary_genotypes = []
                    for g in genotypes:
                        if g == allele1:
                            binary_genotypes.append(0)
                        elif g == allele2:
                            binary_genotypes.append(1)
                        else:
                            binary_genotypes.append(-1)  # Missing data
                    
                    marker_genotypes.append(binary_genotypes)
                    
                except ValueError:
                    continue
        
        # Infer crossover positions
        n_individuals = len(marker_genotypes[0]) if marker_genotypes else 0
        crossover_data = []
        
        for ind in range(n_individuals):
            crossovers = []
            prev_genotype = -1
            prev_position = -1
            
            for marker_idx in range(len(marker_positions)):
                genotype = marker_genotypes[marker_idx][ind]
                position = marker_positions[marker_idx]
                
                if genotype >= 0 and position <= chromosome_length:
                    if prev_genotype >= 0 and genotype != prev_genotype:
                        # Crossover occurred - estimate position
                        crossover_position = 0.5 * (position + prev_position)
                        if crossover_position <= chromosome_length:
                            crossovers.append(crossover_position)
                    
                    prev_genotype = genotype
                    prev_position = position
            
            crossover_data.append(crossovers)
        
        return crossover_data
    
    @staticmethod
    def load_from_positions_file(filename):
        """
        Load crossover data from file with explicit positions.
        
        Expected format: One line per individual, crossover positions separated by spaces.
        """
        crossover_data = []
        
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                positions = []
                parts = line.strip().split()
                
                for part in parts:
                    try:
                        pos = float(part)
                        positions.append(pos)
                    except ValueError:
                        continue
                
                crossover_data.append(positions)
        
        return crossover_data