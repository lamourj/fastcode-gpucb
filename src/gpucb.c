// This version includes the incremental cholesky factorization.

#include "gpucb.h"
#include "mathHelpers.h"
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

double function_baseline(double x, double y) {
    // double t = sin(x) + cos(y);
    double t = -pow(x, 2) - pow(y, 2);
    printf("(C code) Sampled: [%.2lf %.2lf] result %lf \n", x, y, t);
    return t;
}

void learn_baseline(double *X_grid,
           double *K,
           double *L_T,
           bool *sampled,
           int *X,
           double *T,
           int t,
           int maxIter,
           double *mu,
           double *sigma,
           double(*kernel)(double *, double *, double *, double *),
           double beta,
           int n) {
    /*
     * grid_idx = self.argmax_ucb()
    *  self.sample(self.X_grid[grid_idx])
    *  for every point x:
     *  gp_regression()
    *  gp.fit(self.X, self.T)
    *  mu1 = self.mu
     */
    bool debug = true;
    int maxI = 0;
    int maxJ = 0;
    double max = mu[0] + sqrt(beta) * sigma[0];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double currentValue = mu[i * n + j] + sqrt(beta) * sigma[i * n + j];

            if (!sampled[i * n + j] && (currentValue > max)) {
                max = currentValue;
                maxI = i;
                maxJ = j;
            }
        }
    }

    X[2 * t] = maxI;
    X[2 * t + 1] = maxJ;
    sampled[maxI * n + maxJ] = true;
    T[t] = function_baseline(X_grid[maxI * 2 * n + 2 * maxJ], X_grid[maxI * 2 * n + 2 * maxJ + 1]);
    gp_regression_baseline(X_grid, K, L_T, X, T, t, maxIter, kernel, mu, sigma, n); // updating mu and sigma for every x in X_grid
}

double kernel2_baseline(double *x1, double *y1, double *x2, double *y2) {
    // RBF kernel
    double sigma = 1;
    return exp(-((*x1 - *x2) * (*x1 - *x2) + (*y1 - *y2) * (*y1 - *y2)) / (2 * sigma * sigma));
}

void initialize_meshgrid_baseline(double *X_grid, int n, double min, double inc) {
    double x = min;
    for (int i = 0; i < n; i++) {
        double y = min;
        for (int j = 0; j < n; j++) {
            X_grid[i * 2 * n + 2 * j] = y;
            X_grid[i * 2 * n + 2 * j + 1] = x; // With this assignment, meshgrid is the same as python code
            y += inc;
        }
        x += inc;
    }
}

void gpucb_initialized_baseline(double *X_grid,
                       double *K,
                       double *L_T,
                       bool *sampled,
                       int *X,
                       double *T,
                       int maxIter,
                       double *mu,
                       double *sigma,
                       double beta,
                       int n) {
    for (int t = 0; t < maxIter; t++) {
        learn_baseline(X_grid, K, L_T, sampled, X, T, t, maxIter, mu, sigma, kernel2_baseline, beta, n);
    }
}

int gpucb_baseline(int maxIter, int n, double grid_min, double grid_inc) {

    // Allocations
    double       mu[n * n];
    double       sigma[n * n];
    double       X_grid[2 * n * n];
    double       K[maxIter * maxIter];
    double       L_T[maxIter * maxIter];
    double       T[maxIter];
    int          X[2 * maxIter];
    bool         sampled[n * n];
    const double beta = 100;

    // Initializations
    for (int i = 0; i < n * n; i++) {
        sampled[i] = false;
        mu[i] = 0;
        sigma[i] = 0.5;
    }
    initialize_meshgrid_baseline(X_grid, n, grid_min, grid_inc);


    // -------------------------------------------------------------
    //                  Done with initializations
    // -------------------------------------------------------------

    gpucb_initialized_baseline(X_grid, K, L_T, sampled, X, T, maxIter, mu, sigma, beta, n);

    // -------------------------------------------------------------
    //           Done with gpucb; rest is output writing
    // -------------------------------------------------------------

    FILE *f = fopen("mu_c.txt", "w");
    bool printMuConsole = false;
    bool printSigmaConsole = false;
    if (printMuConsole) {
        printf("Mu matrix after training: \n");
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(f, "%lf, ", mu[i * n + j]);
            if (printMuConsole) {
                printf("%.5lf ", mu[i * n + j]);
            }
        }
        fprintf(f, "\n");
        if (printMuConsole) {
            printf("\n");
        }
    }
    fclose(f);

    if (printSigmaConsole) {
        printf("\n\nSigma matrix after training: \n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%.5lf ", sigma[i * n + j]);
            }
            printf("\n");
        }
    }


    return 0;
}