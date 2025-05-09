"""
pso.py

This file implements the Particle Swarm Optimization (PSO) algorithm. The PSO algorithm
is a population-based optimization technique inspired by the social behavior of birds
and fish. It is used to find the global minimum or maximum of a function.

The implementation includes parallel evaluation of particles using Python's 
`ThreadPoolExecutor` to improve performance.

Classes:
    PSO: Implements the PSO algorithm, including particle initialization, evaluation,
         and updating of velocity and position.

Author: Esteban García Muñoz
Date: 13/05/2025
"""
# Import necessary libraries
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Tuple

#Import the Particle class from the particle module
from .particle import Particle

class PSO:
    
    def __init__(self, swarmsize: int, maxiter: int, bounds: Tuple[np.ndarray, np.ndarray],
                  w: float, c1: float, c2: float, function: Callable[[np.ndarray], float]
    ) -> None:
        """ Constructor:

        Args:
            swarmsize: number of particles in the swarm
            maxiter: maximum number of iterations
            bounds: array of tuples with the lower and upper bounds of the search space
            w: inertia weight
            c1: cognitive parameter
            c2: social parameter
            function: function to evaluate the particles
        """
        self.swarmsize: int = swarmsize
        self.maxiter: int = maxiter
        self.bounds: Tuple[np.ndarray, np.ndarray] = bounds
        self.w: float = w
        self.c1: float = c1
        self.c2: float = c2
        self.function: Callable[[np.ndarray], float] = function

    
    def initialize_particles(self)-> List[Particle]:
        """ Initialize the particles in the swarm
    
        Returns: 
            particles: list of particles
        """
        particles: List[Particle] = []

        for _ in range(self.swarmsize):
            particles.append(Particle(bounds=self.bounds))
        
        return particles


    def pso(self):
        """ Perform the Particle Swarm Optimization (PSO) algorithm.

        Returns:
            Tuple[np.ndarray, float]: The best global position and its corresponding value.
        """
        particles: List[Particle] = self.initialize_particles()

        # Historical best global position and value
        gBest: np.ndarray = None
        gBest_value: float = np.inf

        with ThreadPoolExecutor(max_workers=8) as executor:

            # Iterate over the number of iterations
            for _ in range(self.maxiter):

                # Evaluate particles in parallel
                futures = {executor.submit(p.evaluate, self.function): p for p in particles}
                for fut in as_completed(futures):
                    p = futures[fut]
                    value = fut.result()  # Get the result of evaluate()
                    if value < gBest_value:
                        gBest_value = value
                        gBest = p.position.copy()

                # Update velocity and position for each particle in parallel
                executor.map(
                    lambda p: p.update_velocity_and_position(
                        gbest_pos = gBest,
                        w         = self.w,
                        c1        = self.c1,
                        c2        = self.c2,
                        bounds    = self.bounds
                    ),
                    particles
                )
        
        return gBest, gBest_value