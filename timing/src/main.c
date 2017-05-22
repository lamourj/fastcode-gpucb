#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "perf.h"
#include "gpucb3.h"
#include "gpucb.h"


#define N 10


int main() {

    uint64_t cycles_gpucb_baseline, cycles_gpucb_optimized;

    // Execution variables
    const int n = 360; // Meshgrid size
    const int maxIter = 40; // GP-UCB # of iterations
    const double grid_min = -9;
    const double grid_inc = 0.05;

    if(! (n % 4 == 0)) {
        printf("n is not divisible by 4 !!! \n");
        fprintf(stderr, " WARNING: maxIter>=n: maxIter=%d, n=%d", maxIter, n);
        fprintf(stderr, " n is not divisible by 4 !!!, n=%d", n);
    }

    

    // Allocate memory
    double T[maxIter];
    int X[2 * maxIter];
    double X_grid[2 * n * n];
    bool sampled[n * n];
    double mu[n * n];
    double mu_opt[n * n];
    double sigma[n * n];
    const double beta = 10;
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
    for (i = 0; i < N; i += 1) gpucb_initialized_baseline(X_grid, K, L, sampled, X, T, maxIter, mu, sigma, beta, n);

    cycles_gpucb_baseline = cycles_count_stop();


    // Re-initialize matrices
    for (int i = 0; i < n * n; i++) {
        sampled[i] = false;
        mu_opt[i] = 0;
        sigma[i] = 0.5;
    }
    initialize_meshgrid_baseline(X_grid, n, grid_min, grid_inc);
    // warm up the cache
    for (i = 0; i < N; i += 1) gpucb_initialized(X_grid, K, L, sampled, X, T, maxIter, mu_opt, sigma, beta, n);

    
    cycles_count_start();
    for (i = 0; i < N; i += 1) gpucb_initialized(X_grid, K, L, sampled, X, T, maxIter, mu_opt, sigma, beta, n);
    cycles_gpucb_optimized = cycles_count_stop();

    perf_done();

    printf("gpucb baseline: %lf cycles\n", (double) cycles_gpucb_baseline / N);
    printf("gpucb version 3: %lf cycles\n", (double) cycles_gpucb_optimized / N);
    printf("Speedup: %lf\n", (double) cycles_gpucb_baseline / cycles_gpucb_optimized);


    // Search for maximal point:
    double maxBaseline, iBaseline, jBaseline, maxVersion, iVersion, jVersion;
    iBaseline = 0;
    jBaseline = 0;
    iVersion = 0;
    jVersion = 0;
    maxBaseline = mu[0];
    maxVersion = mu_opt[0];


    int diffCounter = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double mui = mu[i * n + j];
            double mu_opti = mu_opt[i * n + j];

            if(mui > maxBaseline) {
                maxBaseline = mui;
                iBaseline = i;
                jBaseline = j;
            }

            if(mu_opti > maxVersion) {
                maxVersion = mu_opti;
                iVersion = i;
                jVersion = j;
            }

            if(mui - mu_opti > 0.01) {
                diffCounter++;
            }
        }
    }


    if(diffCounter > 0) {
        printf("Validation failed on %d/%d elements.\n", diffCounter, n*n);
    }

    int maxBaselineIdx = 2 * (n * iBaseline + jBaseline);
    int maxVersionIdx = 2 * (n * iVersion + jVersion);

    printf("Baseline version: max=%.5lf at [%.5lf %.5lf]\n", maxBaseline, X_grid[maxBaselineIdx], X_grid[maxBaselineIdx+1]);
    printf("Opt version:      max=%.5lf at [%.5lf %.5lf]\n", maxVersion, X_grid[maxVersionIdx], X_grid[maxVersionIdx+1]);

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