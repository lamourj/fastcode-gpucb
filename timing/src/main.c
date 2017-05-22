#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "perf.h"
#include "../../src/gpucb3.h"
#include "../../src/gpucb.h"


#define N 10


int main() {

    uint64_t cycles_gpucb_baseline, cycles_gpucb_optimized;

    // Execution variables
    const int n = 360; // Meshgrid size
    const int maxIter = 40; // GP-UCB # of iterations
    const double grid_min = -9;
    const double grid_inc = 0.05;

    const double beta = 100;

    if (!(n % 4 == 0)) {
        fprintf(stderr, " n is not divisible by 4 !!!, n=%d", n);
    }
    if (maxIter >= n) {
        fprintf(stderr, " WARNING: maxIter>=n: maxIter=%d, n=%d", maxIter, n);
    }



    // Allocate memory

    double *T;
    int *X;
    double *X_grid;
    bool *sampled;
    double *mu;
    double *mu_opt;
    double *sigma;
    double *K;
    double *L;

    T = (double *) malloc(maxIter * sizeof(double));
    X = (int *) malloc(2 * maxIter * sizeof(int));
    X_grid = (double *) malloc(2 * n * n * sizeof(double));
    sampled = (bool *) malloc(n * n * sizeof(bool));
    mu = (double *) malloc(n * n * sizeof(double));
    mu_opt = (double *) malloc(n * n * sizeof(double));
    sigma = (double *) malloc(n * n * sizeof(double));
    K = (double *) malloc(maxIter * maxIter * sizeof(double));
    L = (double *) malloc(maxIter * maxIter * sizeof(double));

    if (T == 0 || X == 0 || X_grid == 0 || sampled == 0 || mu == 0 || mu_opt == 0 || sigma == 0 || K == 0 || L == 0) {
        printf("ERROR: Out of memory\n");
        return 1;
    }



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

            if (mui > maxBaseline) {
                maxBaseline = mui;
                iBaseline = i;
                jBaseline = j;
            }

            if (mu_opti > maxVersion) {
                maxVersion = mu_opti;
                iVersion = i;
                jVersion = j;
            }

            if (mui - mu_opti > 0.01) {
                diffCounter++;
            }
        }
    }


    if (diffCounter > 0) {
        printf("Validation failed on %d/%d elements.\n", diffCounter, n * n);
    }

    int maxBaselineIdx = 2 * (n * iBaseline + jBaseline);
    int maxVersionIdx = 2 * (n * iVersion + jVersion);

    printf("Baseline version: max=%.5lf at [%.5lf %.5lf]\n", maxBaseline, X_grid[maxBaselineIdx],
           X_grid[maxBaselineIdx + 1]);
    printf("Opt version:      max=%.5lf at [%.5lf %.5lf]\n", maxVersion, X_grid[maxVersionIdx],
           X_grid[maxVersionIdx + 1]);

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


    free(T);
    free(X);
    free(X_grid);
    free(sampled);
    free(mu);
    free(mu_opt);
    free(sigma);
    free(K);
    free(L);

    return 0;
}