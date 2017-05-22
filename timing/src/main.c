#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "perf.h"
#include "gpucb3.h"
#include "gpucb.h"


#define N 1


int main() {

    uint64_t cycles_gpucb_baseline, cycles_gpucb_optimized;

    // Execution variables
    const int n = 24; // Meshgrid size
    const int maxIter = 100; // GP-UCB # of iterations
    const double grid_min = -3;
    const double grid_inc = 0.25;

    if(! (n % 4 == 0)) {
        printf("n is not divisible by 4 !!! \n");
    }

    // Allocate memory
    double T[maxIter];
    int X[2 * maxIter];
    double X_grid[2 * n * n];
    bool sampled[n * n];
    double mu[n * n];
    double mu_opt[n * n];
    double sigma[n * n];
    const double beta = 100;
    double K[maxIter * maxIter];
    double L[maxIter * maxIter];


    // Initialize matrices
    for (int i = 0; i < n * n; i++) {
        sampled[i] = false;
        mu[i] = 0;
        sigma[i] = 0.5;
    }


    initialize_meshgrid_baseline(X_grid, n, grid_min, grid_inc);


    int i;
    perf_init();

    // warm up the cache
    for (i = 0; i < N; i += 1) gpucb_initialized_baseline(X_grid, K, L, sampled, X, T, maxIter, mu, sigma, beta, n);

    cycles_count_start();
    for (i = 0; i < N; i += 1)     gpucb_initialized_baseline(X_grid, K, L, sampled, X, T, maxIter, mu, sigma, beta, n);

    cycles_gpucb_baseline = cycles_count_stop();

    // Re-initialize matrices
    for (int i = 0; i < n * n; i++) {
        sampled[i] = false;
        mu_opt[i] = 0;
        sigma[i] = 0.5;
    }
    initialize_meshgrid_baseline(X_grid, n, grid_min, grid_inc);
    // warm up the cache
    for (i = 0; i < N; i += 1) gpucb_initialized(maxIter, n, T, X, X_grid, sampled, mu_opt, sigma, beta);

    cycles_count_start();
    for (i = 0; i < N; i += 1) gpucb_initialized(maxIter, n, T, X, X_grid, sampled, mu_opt, sigma, beta);
    cycles_gpucb_optimized = cycles_count_stop();

    perf_done();

    printf("gpucb baseline: %lf cycles\n", (double) cycles_gpucb_baseline / N);
    printf("gpucb version 3: %lf cycles\n", (double) cycles_gpucb_optimized / N);
    printf("Speedup: %lf\n", (double) cycles_gpucb_baseline / cycles_gpucb_optimized);


    // Validation:
    int diffCounter = 0;
    for(i = 0; i < n * n; i++) {
        int mui = mu[i];
        int mu_opti = mu_opt[i];

        if(mui != mu_opti) {
            diffCounter++;
        }
    }
    if(diffCounter > 0) {
        printf("Validation failed on %d/%d elements.\n", diffCounter, n*n);
    }

    // Save output to file:
    if (false) {
        FILE *f = fopen("mu_c.txt", "w");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                fprintf(f, "%lf, ", mu[i * n + j]);
            }
            fprintf(f, "\n");
        }
        fclose(f);
    }
    return 0;
}