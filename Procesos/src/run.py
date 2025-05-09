# cd "c:\Users\esteb\Desktop\CUNEF\2024-2025\Segundo cuatri\Programación Paralela\Entrega2\Conc"
# python -m src.run
# Barra de progreso con tqdm poner con el prodcto cartesiano!!!! (Esta en parametría)
"""
run.py

This script orchestrates the execution of the Particle Swarm Optimization (PSO) algorithm
using multiprocessing to evaluate multiple parameter combinations in parallel. The results
are saved to a CSV file for further analysis.

The implementation leverages Python's `ProcessPoolExecutor` (via `execute_pso`) to run the PSO
algorithm concurrently across multiple processes, improving performance for large-scale
parameter sweeps.

Modules:
    - parametria: Provides the `execute_pso` function for parallel execution of PSO runs.
    - functions: Contains test functions (e.g., quadratic, rastrigin, ackley) to optimize.

Author: Esteban García Muñoz
Date:   13/05/2025
"""

# Import necessary libraries
import numpy as np
import itertools
import os
from typing import Callable, Tuple, List  

# Import custom modules
from src.PSO.parametria import execute_pso
from src.test_functions.functions import quadratic, rastrigin, ackley

if __name__ == "__main__":

    # Define parameter ranges for the PSO algorithm
    swarmsize: np.ndarray = np.linspace(10, 150, 4, dtype=int) 
    omega: np.ndarray = np.linspace(0.4, 0.9, 4)               
    phip: np.ndarray = np.linspace(1.0, 2.0, 4)                
    phig: np.ndarray = np.linspace(1.0, 2.0, 4)                
    maxiter: np.ndarray = np.linspace(60, 800, 4, dtype=int)   
    dimensions: List[int] = [2, 5, 10]                        
    functions: List[Callable] = [quadratic, rastrigin, ackley]     

    # Define the search space bounds for each dimension
    bounds: Tuple[List[np.ndarray], List[np.ndarray]] = (
        [np.array([-5.12] * dim) for dim in dimensions], [np.array([5.12] * dim) for dim in dimensions]
    )

    # Compute the Cartesian product of all parameter combinations
    combinations: List[Tuple[int, float, float, float, int, int, Callable]] = list(
        itertools.product(swarmsize, omega, phip, phig, maxiter, dimensions, functions)
        )

    # Execute the PSO algorithm for all parameter combinations in parallel
    combinations_df = execute_pso(combinations, bounds, dimensions)

    # Define the output directory and ensure it exists
    output_dir: str = os.path.join(os.path.dirname(__file__),"..", "data")

    # Save the results to a CSV file
    output_file: str = os.path.join(output_dir, "pso_resultsProcesos.csv")
    combinations_df.to_csv(output_file, index=False)

    # Print the location of the saved results
    print(f"Resultados guardados en: {output_file}")


