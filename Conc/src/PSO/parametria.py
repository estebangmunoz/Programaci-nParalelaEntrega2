import pandas as pd
from tqdm import tqdm
import time

from src.PSO.pso import PSO

def execute_pso(combinations, bounds, dimensions):
    """Function to execute PSO for each combination of values and return a DataFrame with the results"""
    results = []

    # Executing the algorithm for each combination of values
    for comb in tqdm(combinations, desc = "Executing combinations"):
        # Storing the values of the combination
        swarmsize, omega, phip, phig, max_iter, dim, function = comb

        # Select the bounds for the current dimension
        current_bounds = [bounds[0][dimensions.index(dim)], bounds[1][dimensions.index(dim)]]

        # Measure the time of execution of the iteration
        start_time = time.time()
        
        # Create instance of the PSO algorithm
        pso = PSO(swarmsize, max_iter, current_bounds, omega, phip, phig, function)
        # Executing the algorithm and storing the results
        bestPosition, bestValue = pso.pso()

        # Calculate the total time of execution
        end_time = time.time()
        total_time = end_time - start_time

        #Adding the results to the list
        results.append([function.__name__, dim, swarmsize, omega, phip, phig, max_iter, bestPosition, bestValue, total_time])

    # Creating a pandas DataFrame with the results
    df = pd.DataFrame(results, columns=['function','dimensions','swarmsize', 'omega', 'phip', 'phig', 'max_iter', 'best_position', 'best_value', 'total_time(s)'])

    return df