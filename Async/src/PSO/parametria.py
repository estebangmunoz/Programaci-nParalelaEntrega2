"""
parametria.py

This file defines the parameters and combinations required to execute the
Particle Swarm Optimization (PSO) algorithm. The parameters are generated
as lists of values and combined using the Cartesian product to explore all
possible configurations.

Author: Esteban García Muñoz
Date: 13/05/2025
"""

# Import necessary libraries
import itertools
import numpy as np
from typing import List, Callable, Tuple

# Import test functions
from src.test_functions.functions import quadratic, rastrigin, ackley

# Parameter definitions
"""
Define the parameter lists to be used by the PSO algorithm.
These parameters must match exactly with those defined in the run.py file
to ensure consistency across experiments.
"""
# Swarm size: number of particles in the search space.
swarmsize: np.ndarray = np.linspace(10, 150, 2, dtype=int) 

# Inertia weight (omega): controls the influence of the previous velocity on the new velocity.
omega: np.ndarray = np.linspace(0.4, 0.9, 2)               

# Cognitive coefficient (phip): weight of the attraction to the personal best position.
phip: np.ndarray = np.linspace(1.0, 2.0, 2)     

# Social coefficient (phig): weight of the attraction to the global best position.
phig: np.ndarray = np.linspace(1.0, 2.0, 2)                

# Maximum number of iterations: upper limit of iterations for the algorithm.
maxiter: np.ndarray = np.linspace(60, 800, 2, dtype=int)

# Search space dimensions: number of variables to optimize.
dimensions: List[int] = [2, 3]                        

# Test functions: objective functions to be optimized by the algorithm.
functions: List[Callable] = [quadratic, rastrigin, ackley]     

# Generate parameter combinations
"""
Use the Cartesian product to generate all possible combinations of the
parameters defined above. This allows for an exhaustive exploration of the
PSO algorithm's configuration space.
"""
combinations: List[Tuple[int, float, float, float, int, int, Callable]] = list(itertools.product(
    swarmsize,
    omega,
    phip,
    phig,
    maxiter,
    dimensions,
    functions
))