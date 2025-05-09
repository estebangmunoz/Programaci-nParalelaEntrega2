/*
pso_openmp.c

This file implements the Particle Swarm Optimization (PSO) algorithm using OpenMP for parallelism.
The PSO algorithm is a population-based optimization technique inspired by the social behavior
of birds and fish. It is used to find the global minimum or maximum of a function.

The implementation evaluates multiple parameter combinations in parallel and writes the results
to a CSV file for further analysis.

Features:
    - Parallel evaluation of particles using OpenMP.
    - Support for multiple benchmark functions (quadratic, rastrigin, ackley).
    - Configurable parameters for swarm size, dimensions, iterations, and coefficients.

Structures:
    - Particle: Represents an individual particle in the PSO algorithm.
    - PSOResult: Stores the results of a single PSO run.

Functions:
    - quad: Implements the quadratic benchmark function.
    - rastrigin: Implements the Rastrigin benchmark function.
    - ackley: Implements the Ackley benchmark function.
    - run_pso: Executes the PSO algorithm for a given set of parameters.
    - main: Orchestrates the parameter sweep and writes results to a CSV file.

Author: Esteban García Muñoz
Date:   13/05/2025
*/

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <time.h>

// Define constants for benchmark function bounds
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

#define LOWER_B -5.12
#define UPPER_B  5.12

// Structure representing a particle in the PSO algorithm
typedef struct {
    double *position;
    double *velocity;
    double *pbest_pos;
    double  pbest_val;
} Particle;

// Structure to store the results of a PSO run
typedef struct {
    double best_val;
    double *best_pos;
    int    best_iter;
    double duration;
} PSOResult;

// Function prototypes for benchmark functions
double quad(const double *x, int dim);
double rastrigin(const double *x, int dim);
double ackley(const double *x, int dim);

// Array of benchmark functions and their names
typedef double (*bench_f)(const double*, int);
bench_f funcs[] = { quad, rastrigin, ackley };
const char *func_names[] = { "quadratic", "rastrigin", "ackley" };
int n_funcs = sizeof(funcs) / sizeof(funcs[0]);

// Quadratic benchmark function
double quad(const double *x, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) sum += x[i] * x[i];
    return sum;
}

// Rastrigin benchmark function
double rastrigin(const double *x, int dim) {
    double sum = 10.0 * dim;
    for (int i = 0; i < dim; i++) {
        sum += x[i] * x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
    }
    return sum;
}

// Ackley benchmark function
double ackley(const double *x, int dim) {
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < dim; i++) {
        sum1 += x[i] * x[i];
        sum2 += cos(2.0 * M_PI * x[i]);
    }
    sum1 = -0.2 * sqrt(sum1 / dim);
    sum2 = -1.0 * sum2 / dim;
    return 20.0 + M_E - 20.0 * exp(sum1) - exp(sum2);
}

// Executes the PSO algorithm for a given set of parameters
PSOResult run_pso(int swarmsize, int dim, int maxiter,
                  double w, double c1, double c2,
                  double lower_b, double upper_b,
                  bench_f func) {
    // Initialize particles
    Particle *particles = malloc(swarmsize * sizeof(Particle));
    for (int i = 0; i < swarmsize; i++) {
        particles[i].position  = malloc(dim * sizeof(double));
        particles[i].velocity  = malloc(dim * sizeof(double));
        particles[i].pbest_pos = malloc(dim * sizeof(double));
        particles[i].pbest_val = DBL_MAX;
        for (int j = 0; j < dim; j++) {
            double range = upper_b - lower_b;
            particles[i].position[j] = lower_b + ((double)rand() / RAND_MAX) * range;
            particles[i].velocity[j] = ((double)rand() / RAND_MAX - 0.5) * range;
        }
    }
    double gbest_val = DBL_MAX;
    double *gbest_pos = malloc(dim * sizeof(double));
    int best_iter = 0;
    int i, j;

    double t_start = omp_get_wtime();
    
    // Initial evaluation
    #pragma omp parallel
    {
        double local_min = DBL_MAX;
        #pragma omp for schedule(static)
        for (i = 0; i < swarmsize; i++) {
            double val = func(particles[i].position, dim);
            if (val < particles[i].pbest_val) {
                particles[i].pbest_val = val;
                memcpy(particles[i].pbest_pos, particles[i].position, dim * sizeof(double));
            }
            if (val < local_min) local_min = val;
        }
        #pragma omp critical
        if (local_min < gbest_val) {
            gbest_val = local_min;
            best_iter = 0;
        }
    }
    // Capture gbest position
    for (i = 0; i < swarmsize; i++) {
        if (particles[i].pbest_val == gbest_val) {
            memcpy(gbest_pos, particles[i].pbest_pos, dim * sizeof(double));
            break;
        }
    }

    // Main PSO iterations
    for (int iter = 1; iter <= maxiter; iter++) {
        #pragma omp parallel
        {
            double local_min = gbest_val;
            #pragma omp for schedule(static)
            for (i = 0; i < swarmsize; i++) {
                double val = func(particles[i].position, dim);
                if (val < particles[i].pbest_val) {
                    particles[i].pbest_val = val;
                    memcpy(particles[i].pbest_pos, particles[i].position, dim * sizeof(double));
                }
                if (val < local_min) local_min = val;
            }
            #pragma omp critical
            if (local_min < gbest_val) {
                gbest_val = local_min;
                best_iter = iter;
            }
        }
        
        for (i = 0; i < swarmsize; i++) {
            if (particles[i].pbest_val == gbest_val) {
                memcpy(gbest_pos, particles[i].pbest_pos, dim * sizeof(double));
                break;
            }
        }
        // Update particle positions and velocities
        #pragma omp parallel for schedule(static) private(i,j)
        for (i = 0; i < swarmsize; i++) {
            for (j = 0; j < dim; j++) {
                double r1 = (double)rand() / RAND_MAX;
                double r2 = (double)rand() / RAND_MAX;
                double cog = c1 * r1 * (particles[i].pbest_pos[j] - particles[i].position[j]);
                double soc = c2 * r2 * (gbest_pos[j]    - particles[i].position[j]);
                particles[i].velocity[j] = w * particles[i].velocity[j] + cog + soc;
                particles[i].position[j] += particles[i].velocity[j];
                if (particles[i].position[j] < lower_b) particles[i].position[j] = lower_b;
                if (particles[i].position[j] > upper_b) particles[i].position[j] = upper_b;
            }
        }
    }
    
    double t_end = omp_get_wtime();
    double duration = t_end - t_start;

    // Free particle memory
    for (i = 0; i < swarmsize; i++) {
        free(particles[i].position);
        free(particles[i].velocity);
        free(particles[i].pbest_pos);
    }
    free(particles);

    PSOResult res = { gbest_val, gbest_pos, best_iter, duration };
    return res;
}

// Main function to orchestrate parameter sweeps
int main() {
    // Parameter ranges
    int swarm_sizes[] = {100, 250, 500, 1000, 2000};
    int dims[]        = {2, 5, 9, 10, 15};
    int maxiters[]    = {60, 150, 250, 500, 1000};
    double ws[]       = {0.4, 0.5, 0.6, 0.8, 1.0};
    double c1s[]      = {1.0, 1.3, 1.5, 1.7, 2.0};
    double c2s[]      = {1.0, 1.3, 1.5, 1.7, 2.0};

    // Calculate array sizes
    int n_sw   = sizeof(swarm_sizes) / sizeof(swarm_sizes[0]);
    int n_dim  = sizeof(dims)        / sizeof(dims[0]);
    int n_mi   = sizeof(maxiters)    / sizeof(maxiters[0]);
    int n_w    = sizeof(ws)          / sizeof(ws[0]);
    int n_c1   = sizeof(c1s)         / sizeof(c1s[0]);
    int n_c2   = sizeof(c2s)         / sizeof(c2s[0]);

    // Open CSV file for writing
    FILE *fp = fopen("pso_resultsMP.csv", "w");
    if (!fp) {
        perror("Error al crear pso_resultsMP.csv");
        return 1;
    }
    fprintf(fp, "function,swarmsize,dim,maxiter,w,c1,c2,duration,best_iter,best_val,best_pos\n");

    int i_sw, i_d, i_m, i_wv, i_c1, i_c2, k, j;

    // Perform parameter sweeps
    for (k = 0; k < n_funcs; k++) {
        for (i_sw = 0; i_sw < n_sw; i_sw++) {
            for (i_d = 0; i_d < n_dim; i_d++) {
                for (i_m = 0; i_m < n_mi; i_m++) {
                    for (i_wv = 0; i_wv < n_w; i_wv++) {
                        for (i_c1 = 0; i_c1 < n_c1; i_c1++) {
                            for (i_c2 = 0; i_c2 < n_c2; i_c2++) {
                                int sw = swarm_sizes[i_sw];
                                int d  = dims[i_d];
                                int mi = maxiters[i_m];
                                double wv  = ws[i_wv];
                                double c1v = c1s[i_c1];
                                double c2v = c2s[i_c2];
                                PSOResult res = run_pso(
                                    sw, d, mi, wv, c1v, c2v,
                                    LOWER_B, UPPER_B, funcs[k]
                                );
                                // Escribir CSV con best_pos entre comillas
                                fprintf(fp,
                                    "%s,%d,%d,%d,%.2f,%.2f,%.2f,%.6f,%d,%g,\"",
                                    func_names[k],sw,d,mi,
                                    wv,c1v,c2v,
                                    res.duration,res.best_iter,res.best_val
                                );
                                for (j = 0; j < d; j++) {
                                    fprintf(fp, "%g%s",
                                        res.best_pos[j],
                                        (j<d-1)?";":""
                                    );
                                }
                                fprintf(fp, "\"\n");
                                free(res.best_pos);
                            }
                        }
                    }
                }
            }
        }
    }
    fclose(fp);
    return 0;
}
