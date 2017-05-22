#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "gpucb.h" // CHANGE TO gpucb1.h for newer version

int main(int argc, char *argv[]) {
    // Execution variables
    const int n = 240; // Meshgrid size
    const int maxIter = atoi(argv[1]); // GP-UCB # of iterations
    if(maxIter >= n) {
        fprintf(stderr, " WARNING: maxIter>=n: maxIter=%d, n=%d", maxIter, n);
    }
    const double grid_min = -6;
    const double grid_inc = 0.05;

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


    // Baseline
    double K[maxIter*maxIter];
    double L[maxIter*maxIter];
    initialize_meshgrid_baseline(X_grid, n, grid_min, grid_inc);
    gpucb_initialized_baseline(X_grid, K, L, sampled, X, T, maxIter, mu, sigma, beta, n);


    // Find maximum point:
    int maxI = 0;
    int maxJ = 0;
    double max = mu[0];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double currentValue = mu[i * n + j];
            if (currentValue > max) {
                max = currentValue;
                maxI = i;
                maxJ = j;
            }
        }
    }
    double maxX = X_grid[maxI * 2 * n + 2 * maxJ];
    double maxY = X_grid[maxI * 2 * n + 2 * maxJ + 1];
    printf("\nMaximal point found by C code is %f at [%f %f]\n\n", max, maxX, maxY);

    // Save output to file:
    if (true) {
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