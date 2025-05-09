"""
run.py

This file orchestrates the execution of the Particle Swarm Optimization (PSO) algorithm
using asynchronous programming to evaluate multiple parameter combinations in parallel.
The results are saved to a CSV file for further analysis.

The implementation leverages Python's `asyncio` and `ProcessPoolExecutor` to run the PSO
algorithm concurrently across multiple processes, improving performance for large-scale
parameter sweeps.

Functions:
    _run_one: Executes the PSO algorithm for a single set of parameters asynchronously.
    main: Manages the asynchronous execution of all parameter combinations and writes results to a CSV file.

Author: Esteban García Muñoz
Date: 13/05/2025
"""
# Import necessary libraries
import asyncio
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Tuple, Any, List

# Import the PSO class and parameter combinations
from src.PSO.pso import PSO
from src.PSO.parametria import combinations

async def _run_one(
    params: Tuple[int, float, float, float, int, int, Any], executor: ProcessPoolExecutor) -> Tuple[
        str, int, int, int, float, float, float, float, List[float], float]:
    """ Executes the PSO algorithm for a single set of parameters asynchronously.

    Args:
        params (Tuple[int, float, float, float, int, int, Any]): A tuple containing the parameters for the PSO algorithm.
        executor (ProcessPoolExecutor): Executor to run the PSO algorithm in a separate process.

    Returns:
        Tuple[str, int, int, int, float, float, float, float, List[float], float]: 
            A tuple containing the function name, swarm size, dimensions, max iterations, 
            omega, c1, c2, duration, best position, and best value.
    """
    # Unpack parameters
    sw, omega, phip, phig, maxiter, dim, func = params

    # Define the lower and upper bounds for the search space
    lower = [-5.12] * dim
    upper = [ 5.12] * dim

    # Create a PSO instance with the given parameters
    pso = PSO(
        swarmsize=sw,
        maxiter=maxiter,
        bounds=(lower, upper),
        w=omega,
        c1=phip,
        c2=phig,
        function=func
    )

    # Get the current event loop
    loop = asyncio.get_running_loop()

    # Run the PSO algorithm in a separate process and measure the time taken
    start_time = time.perf_counter()
    best_pos, best_val = await loop.run_in_executor(executor, pso.pso)
    duration = time.perf_counter() - start_time

    # Return the results as a tuple
    return (
        func.__name__, sw, dim, maxiter,
        omega, phip, phig,
        duration, best_pos, best_val
    )

async def main() -> None:
    """ Manages the asynchronous execution of all parameter combinations and writes results to a CSV file. """
    # Ensure the output directory exists
    os.makedirs('data', exist_ok=True)

    # Define the maximum number of workers for parallel execution
    max_workers = 8

    # Use a ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of tasks for each parameter combination
        tasks = [asyncio.create_task(_run_one(p, executor)) for p in combinations]

        # Initialize an empty list to store results
        results: List[Tuple[str, int, int, int, float, float, float, float, List[float], float]] = []

        # Process tasks as they complete, with a progress bar
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="PSO async"):
            res = await coro
            results.append(res)

    # Define the path for the output CSV file
    csv_path = os.path.join('data', 'pso_resultsAsync.csv')

    # Write the results to the CSV file
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the header row
        writer.writerow([
            "function", "swarmsize", "dim", "maxiter",
            "omega", "c1", "c2", "total_time", "best_pos", "best_val"
        ])

        # Write each result row
        for func_name, sw, dim, mi, wv, c1v, c2v, dur, best_pos, best_val in results:
            writer.writerow([
                func_name, sw, dim, mi,
                wv, c1v, c2v,
                dur, best_pos, best_val
            ])

if __name__ == "__main__":
    asyncio.run(main())

