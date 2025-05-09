# cd "c:\Users\esteb\Desktop\CUNEF\2024-2025\Segundo cuatri\Programación Paralela\Entrega2\Conc"
# python -m src.run
# Barra de progreso con tqdm poner con el prodcto cartesiano!!!! (Esta en parametría)
"""
run.py

This script orchestrates the execution of the Particle Swarm Optimization (PSO) algorithm
using multithreading to evaluate multiple parameter combinations in parallel. The results
are saved to a CSV file for further analysis.

The implementation leverages Python's `ThreadPoolExecutor` (via `execute_pso`) to run the PSO
algorithm concurrently across multiple threads, improving performance for large-scale
parameter sweeps.

Modules:
    - particle: Defines the `Particle` class for PSO.
    - pso: Implements the PSO algorithm.
    - parametria: Provides the `execute_pso` function for parallel execution of PSO runs.
    - functions: Contains test functions (e.g., quadratic, rastrigin, ackley) to optimize.

Author: Esteban García Muñoz
Date:   13/05/2025
"""

# Import necessary libraries
import numpy as np
import itertools
import os
from typing import Callable, Tuple
import pandas as pd

# Import  modules
from src.PSO.parametria import execute_pso
from src.test_functions.functions import quadratic, rastrigin, ackley

if __name__ == "__main__":

    # Define parameter ranges for the PSO algorithm
    swarmsize: np.ndarray = np.linspace(10, 150, 4, dtype=int)  
    omega: np.ndarray = np.linspace(0.4, 0.9, 4)               
    phip: np.ndarray = np.linspace(1.0, 2.0, 4)                
    phig: np.ndarray = np.linspace(1.0, 2.0, 4)                
    maxiter: np.ndarray = np.linspace(60, 800, 4, dtype=int)   
    dimensions: list[int] = [2, 5, 10]                        
    functions: list[Callable] = [quadratic, rastrigin, ackley]     


    # Define the search space bounds for each dimension
    bounds: Tuple[list[np.ndarray], list[np.ndarray]] = (
        [np.array([-5.12] * dim) for dim in dimensions], 
        [np.array([5.12] * dim) for dim in dimensions]
    )

    #Calculating the cartesian product of the arrays
    combinations: list[Tuple[int, float, float, float, int, int, Callable]] = list(
        itertools.product(swarmsize, omega, phip, phig, maxiter, dimensions, functions)
        )

    # Execute the PSO algorithm for all parameter combinations in parallel
    combinations_df: pd.DataFrame = execute_pso(combinations, bounds, dimensions)

    #Saving the csv file in \doc folder
    output_dir: str = os.path.join(os.path.dirname(__file__),"..", "data")


    output_file: str = os.path.join(output_dir, "pso_resultsHilos.csv")
    combinations_df.to_csv(output_file, index=False)

    print(f"Resultados guardados en: {output_file}")


