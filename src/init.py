"""
Crossover Interference Analysis Package

This package implements the extended gamma-sprinkled (GS) model for analyzing
crossover interference in genetic recombination data. The model accounts for
three types of crossover interference: positive, negative, and no interference.

Citation:
Sapielkin, S., Frenkel, Z., Privman, E., & Korol, A.B. (2025). 
Statistical analysis and simulations that account for simultaneous effects 
of positive, negative, and no crossover interference in multilocus 
recombination data.
"""

from .gamma_distributions import GammaDistribution
from .mixture_model import MixtureGammaModel, Model
from .statistical_tests import StatisticalTests
from .data_processing import PopulationStatistics

__version__ = "1.0.0"
__author__ = "Shaul Sapielkin, Zeev Frenkel"
__email__ = "ssapielkin@campus.haifa.ac.il"

__all__ = [
    'GammaDistribution',
    'MixtureGammaModel', 
    'Model',
    'StatisticalTests',
    'PopulationStatistics'
]