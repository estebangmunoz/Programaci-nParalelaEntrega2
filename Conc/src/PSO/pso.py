import numpy as np

from .particle import Particle

class PSO:
    
    def __init__(self, swarmsize, maxiter, bounds, w, c1, c2, function):
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
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.function = function

    
    def initialize_particles(self):
        """ Initialize the particles in the swarm
    
        Returns: 
            particles: list of particles
        """
        particles = []

        for _ in range(self.swarmsize):
            particles.append(Particle(bounds=self.bounds))
        
        return particles
    
    def _update_velocity(self, particle, gBest):
        """Update the velocity of the particle

        Args:
            particle: The particle to update
            gBest: The global best position
        """
        # Generar r1 y r2 de una distribución uniforme entre 0 y 1
        r1 = np.random.rand(len(particle.position))
        r2 = np.random.rand(len(particle.position))

        # Calcular la nueva velocidad
        particle.velocity = (
            self.w * particle.velocity
            + self.c1 * r1 * (particle.pBest[-1] - particle.position)
            + self.c2 * r2 * (gBest - particle.position)
        )


    def pso(self):
        particles = self.initialize_particles()

        # Historical best global position and value
        gBest = None
        gBest_value = np.inf

        # Encontrar la mejor posición inicial
        for particle in particles:
            value = particle.evaluate(self.function)
            if value < gBest_value:
                gBest_value = value
                gBest = particle.position.copy()

        # Iterate over the number of iterations
        for _ in range(self.maxiter):

            for particle in particles:
                # Evaluate the particle in the function
                value = particle.evaluate(self.function)

                # Update the global best position and value if necessary
                if value < gBest_value:
                    gBest_value = value
                    gBest = particle.position.copy()


            # Iterate over the particles
            for particle in particles:
                # Update the velocity of the particle
                self._update_velocity(particle, gBest)

                # Update the position of the particle
                particle.position = particle.position + particle.velocity

                # If the new position is outside the bounds, put it back in the bounds
                particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])
        
        return gBest, gBest_value