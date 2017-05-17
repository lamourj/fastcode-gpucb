#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "perf.h"
#include "gpucb.h"

#define N 1000



int main () {
    
    uint64_t cycles_gpucb;

    // Execution variables
    const int n = 24; // Meshgrid size
    const int maxIter = 6; // GP-UCB # of iterations
    const double grid_min = -3;
    const double grid_inc = 0.25;

    // Allocate memory
    double T[maxIter];
    int X[2 * maxIter];
    double X_grid[2 * n * n];
    bool sampled[n * n];
    double mu[n * n];
    double sigma[n * n];
    const double beta = 100;

    // Initialize matrices
    for (int i = 0; i < n * n; i++) {
        sampled[i] = false;
        mu[i] = 0;
        sigma[i] = 0.5;
    }


    initialize_meshgrid(X_grid, n, grid_min, grid_inc);


    int i;
    perf_init ();
    // warm up the cache
    for (i = 0; i < N; i += 1) gpucb_initialized(maxIter, n, T, X, X_grid, sampled, mu, sigma, beta);
    cycles_count_start();
    for (i = 0; i < N; i += 1) gpucb_initialized(maxIter, n, T, X, X_grid, sampled, mu, sigma, beta);;
    cycles_gpucb = cycles_count_stop();

    perf_done();
    printf("gpucb  : %lf cycles\n", (double) cycles_gpucb / N);

    
    // Save output to file:
    if(true) {
        FILE *f = fopen("mu_timed.txt", "w");
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