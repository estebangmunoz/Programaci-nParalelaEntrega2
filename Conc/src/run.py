# cd "c:\Users\esteb\Desktop\CUNEF\2024-2025\Segundo cuatri\Programación Paralela\Entrega2\Conc"
# python -m src.run
# Barra de progreso con tqdm poner con el prodcto cartesiano!!!! (Esta en parametría)
import numpy as np
import itertools
import os

from src.PSO.particle import Particle
from src.PSO.pso import PSO
from src.PSO.parametria import execute_pso
from src.test_functions.functions import quadratic, rastrigin, ackley

if __name__ == "__main__":


    # Estudiar parametría
    #Defining arrays with values for the different attributes
    swarmsize = np.linspace(10, 150, 4, dtype=int)  # 5 values between 10 and 150
    omega = np.linspace(0.4, 0.9, 4)               # 5 values between 0.4 and 0.9
    phip = np.linspace(1.0, 2.0, 4)                # 5 values between 1.0 and 2.0
    phig = np.linspace(1.0, 2.0, 4)                # 5 values between 1.0 and 2.0
    maxiter = np.linspace(60, 800, 4, dtype=int)   # 5 values between 60 and 800
    dimensions = [2, 5, 10]                        # Dimensions remain the same
    functions = [quadratic, rastrigin, ackley]     # Functions remain the same

    # Definir los límites del espacio de búsqueda 
    bounds = [np.array([-5.12] * dim) for dim in dimensions], [np.array([5.12] * dim) for dim in dimensions]


    #Calculating the cartesian product of the arrays
    combinations = list(itertools.product(swarmsize, omega, phip, phig, maxiter, dimensions, functions))

    combinations_df = execute_pso(combinations, bounds, dimensions)

    #Saving the csv file in \doc folder
    output_dir = os.path.join(os.path.dirname(__file__),"..", "data")

    output_file = os.path.join(output_dir, "pso_results.csv")
    combinations_df.to_csv(output_file, index=False)

    print(f"Resultados guardados en: {output_file}")


