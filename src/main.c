#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "gpucb.h" // CHANGE TO gpucb1.h for newer version

int main(int argc, char *argv[]) {
    // Execution variables
    const int N = 480; // Meshgrid size
    const int I = atoi(argv[1]); // GP-UCB # of iterations
//
    if (!(N % 4 == 0)) {
        fprintf(stderr, " n is not divisible by 4 !!!, n=%d", N);
    }
    if (I >= N) {
        fprintf(stderr, " WARNING: maxIter>=n: maxIter=%d, n=%d", I, N);
    }
//
//    const double grid_min = -6;
//    const double grid_inc = 0.025;
//
//    const double beta = 100;
//
//    // Allocate memory
//    double *T;
//    int *X;
//    double *X_grid;
//    bool *sampled;
//    double *mu;
//    double *sigma;
//    double *K;
//    double *L;
//
//    T = (double *) malloc(maxIter * sizeof(double));
//    X = (int *) malloc(2 * maxIter * sizeof(int));
//    X_grid = (double *) malloc(2 * n * n * sizeof(double));
//    sampled = (bool *) malloc(n * n * sizeof(bool));
//    mu = (double *) malloc(n * n * sizeof(double));
//    sigma = (double *) malloc(n * n * sizeof(double));
//    K = (double *) malloc(maxIter * maxIter * sizeof(double));
//    L = (double *) malloc(maxIter * maxIter * sizeof(double));
//
//    if (T == 0 || X == 0 || X_grid == 0 || sampled == 0 || mu == 0 || sigma == 0 || K == 0 || L == 0) {
//        printf("ERROR: Out of memory\n");
//        return 1;
//    }
//
//
//
//    // Initialize matrices
//    for (int i = 0; i < n * n; i++) {
//        sampled[i] = false;
//        mu[i] = 0;
//        sigma[i] = 0.5;
//    }
    initialize(I, N);

    // Baseline
    run();
    //initialize_meshgrid_baseline(X_grid, n, grid_min, grid_inc);
    //gpucb_initialized_baseline(X_grid, K, L, sampled, X, T, maxIter, mu, sigma, beta, n);


    // Find maximum point:
    int maxI = 0;
    int maxJ = 0;
    double max = mu_[0];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double currentValue = mu_[i * N + j];
            if (currentValue > max) {
                max = currentValue;
                maxI = i;
                maxJ = j;
            }
        }
    }
    double maxX = X_grid_[maxI * 2 * N + 2 * maxJ];
    double maxY = X_grid_[maxI * 2 * N + 2 * maxJ + 1];
    printf("\nMaximal point found by C code is %f at [%f %f]\n\n", max, maxX, maxY);

    // Save output to file:
    if (true) {
        FILE *f = fopen("mu_c.txt", "w");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(f, "%lf, ", mu_[i * N + j]);
            }
            fprintf(f, "\n");
        }
        fclose(f);
    }
    clean();


    return 0;
}