"""
parametria.py

This file defines the execution of the Particle Swarm Optimization (PSO) algorithm
for multiple parameter combinations using multiprocessing. The results are returned
as a pandas DataFrame for further analysis.

The implementation leverages Python's `ProcessPoolExecutor` to run the PSO algorithm
in parallel, improving performance for large-scale parameter sweeps.

Functions:
    _run_one: Executes the PSO algorithm for a single set of parameters.
    execute_pso: Manages the parallel execution of multiple PSO runs and returns results.

Author: Esteban García Muñoz
Date:   13/05/2025
"""

# Import necessary libraries
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
from typing import List, Tuple, Callable, Any

# Import the PSO class
from src.PSO.pso import PSO


def _run_one(
        comb: Tuple[int, float, float, float, int, int, Callable[[Any], float]], 
        bounds: Tuple[List[List[float]], List[List[float]]], 
        dimensions: List[int]
) -> List[Any]:
    """
    Executes a PSO for the given parameter combination `comb`, using the
    appropriate bounds extracted from `bounds` and `dimensions`.

    Args:
        comb (Tuple[int, float, float, float, int, int, Callable[[Any], float]]): 
            A tuple containing the parameters for the PSO algorithm.
        bounds (Tuple[List[List[float]], List[List[float]]]): 
            A tuple containing the lower and upper bounds for each dimension.
        dimensions (List[int]): List of available dimensions.

    Returns:
        List[Any]: A list containing the results of the PSO run, including the
        function name, dimension, swarm size, parameters, best position, best value, and execution time.
    """
    swarmsize, omega, phip, phig, max_iter, dim, func = comb

    # Find the index of the current dimension in the dimensions list
    idx = dimensions.index(dim)

    # Extract the pre-calculated lower and upper bounds for this dimension
    lower = bounds[0][idx]   
    upper = bounds[1][idx]  
    bounds_dim = (lower, upper)

    # Instantiate and execute the PSO
    pso = PSO(
        swarmsize = swarmsize,
        maxiter   = max_iter,
        bounds    = bounds_dim,
        w         = omega,
        c1        = phip,
        c2        = phig,
        function  = func
    )

    # Measure execution time
    t0 = time.time()
    best_pos, best_val = pso.pso()
    total_time = time.time() - t0

    # Return the results as a list
    return [
        func.__name__, dim, swarmsize,
        omega, phip, phig, max_iter,
        best_pos, best_val, total_time
    ]

def execute_pso(
        combinations: List[Tuple[int, float, float, float, int, int, Callable[[Any], float]]], 
        bounds: Tuple[List[List[float]], List[List[float]]], 
        dimensions: List[int]
) -> pd.DataFrame:
    """
    Launches multiple PSO runs in parallel (one for each parameter combination)
    and returns a DataFrame with all the results.

    Args:
        combinations (List[Tuple[int, float, float, float, int, int, Callable[[Any], float]]]): 
            List of parameter combinations for the PSO algorithm.
        bounds (Tuple[List[List[float]], List[List[float]]]): 
            A tuple containing the lower and upper bounds for each dimension.
        dimensions (List[int]): List of available dimensions.

    Returns:
        pd.DataFrame: A DataFrame containing the results of all PSO runs.
    """

    # Define the column names for the results DataFrame
    columns = [
        "function", "dim", "swarmsize", "omega",
        "phip", "phig", "max_iter", "best_pos", "best_val", "total_time"
    ]

    # Create a partial function to pass bounds and dimensions to _run_one
    worker = partial(_run_one, bounds=bounds, dimensions=dimensions)

    # Create the ProcessPoolExecutor and map _run_one over each parameter combination
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Use tqdm to display a progress bar
        results = list(
            tqdm(
                executor.map(worker, combinations),
                total=len(combinations),
                desc="PSO parametría"
            )
        )

    # Build the DataFrame with the returned rows
    return pd.DataFrame(results, columns=columns)

