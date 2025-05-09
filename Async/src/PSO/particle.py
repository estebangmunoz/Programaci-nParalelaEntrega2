"""
particle.py

This file defines the `Particle` class, which represents an individual particle in the 
Particle Swarm Optimization (PSO) algorithm. Each particle has a position, velocity, 
and personal best value, and it updates its state based on its own experience and the 
global best position found by the swarm.

The `Particle` class includes methods for:
    - Initializing the particle's position and velocity within given bounds.
    - Evaluating the particle's position using a given objective function.
    - Updating the particle's velocity and position based on the PSO algorithm.

Classes:
    Particle: Represents a particle in the PSO algorithm.

Author: Esteban García Muñoz
Date: 13/05/2025
"""

# # Import necessary libraries
import numpy as np
from typing import Callable, List, Tuple

class Particle:
    
    def __init__(self, bounds: Tuple[np.ndarray, np.ndarray]) -> None:
        """ Constructor:

        Args:
            bounds: array of tuples with the lower and upper bounds of the search space
        """

        # Check if the bounds are correctly defined
        if len(bounds) != 2 or len(bounds[0]) != len(bounds[1]):
            raise ValueError("Bounds must be a tuple of two arrays with the same length.")
        
        # Generate a random position within the bounds and the dimension of the search space
        self.position: np.ndarray = np.random.uniform(bounds[0], bounds[1], len(bounds[0]))

        # Generate a random velocity within the bounds and the dimension of the search space
        self.velocity: np.ndarray = np.random.uniform(-1, 1, len(self.position))

        # Initialize the historical best position and value
        self.pBest: List[np.ndarray] = [self.position.copy()]
        
        # Initialize the best personal value with np.inf so that the first evaluation is always better
        self.pBest_value: float = np.inf

        # Initialize the bounds
        self.bounds: Tuple[np.ndarray, np.ndarray] = bounds

    
    def evaluate(self, function: Callable[[np.ndarray], float]) -> float:
        """ Evaluate the particle in the function

        Args:
            function: function to evaluate the particle
    
        Returns:
            value: value of the function evaluated in the particle
    
        """
        # Ensure the position is within bounds
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

        # Evaluate the particle in the function
        value: float = function(self.position)

        # Update the best personal value and position
        if value < self.pBest_value:
            self.pBest_value = value
            # Add the current position to the historical best
            self.pBest.append(self.position.copy())   

        return value
    
    def update_velocity_and_position(
            self, gbest_pos: np.ndarray, w: float, c1: float, c2: float, 
            bounds: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Updates the particle's velocity and position based on the PSO algorithm.

        Args:
            gbest_pos (np.ndarray): Global best position found by the swarm.
            w (float): Inertia weight.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
            bounds (Tuple[np.ndarray, np.ndarray]): Tuple containing the lower and upper bounds.
        """

        # Select the personal best position
        pBest_pos: np.ndarray = self.pBest[-1]

        # Random coefficients
        r1: np.ndarray = np.random.rand(len(self.position))
        r2: np.ndarray = np.random.rand(len(self.position))

        # Compute cognitive and social components
        cognitive: np.ndarray = c1 * r1 * (pBest_pos - self.position)
        social: np.ndarray    = c2 * r2 * (gbest_pos   - self.position)

        # 1) Update velocity
        self.velocity = w * self.velocity + cognitive + social

        # 2) Update position
        self.position += self.velocity

        # 3) Clamp position to bounds
        lower, upper = bounds
        self.position = np.clip(self.position, lower, upper)