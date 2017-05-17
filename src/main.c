#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "gpucb.h" // CHANGE TO gpucb1.h for newer version

int main(int argc, char *argv[]) {
    // Execution variables
    const int n = 24; // Meshgrid size
    const int maxIter = atoi(argv[1]); // GP-UCB # of iterations
    if(maxIter >= n) {
        fprintf(stderr, " WARNING: maxIter>=n: maxIter=%d, n=%d", maxIter, n);
    }
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


    // -------------------------------------------------------------------------------
    //                          OLD VERSION
    initialize_meshgrid_baseline(X_grid, n, grid_min, grid_inc);
    gpucb_initialized_baseline(maxIter, n, T, X, X_grid, sampled, mu, sigma, beta);
    // -------------------------------------------------------------------------------


    // For new version, call this:
    // initialize_meshgrid_baseline(X_grid, n, grid_min, grid_inc);
    // Then call gpucb_initialized with appropriate arguments (signature changed).

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