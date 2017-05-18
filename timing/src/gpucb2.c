// Baseline version.

#include "gpucb.h"
#include "mathHelpers2.h"
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

double function(double x, double y) {
    // double t = sin(x) + cos(y);
    double t = -pow(x, 2) - pow(y, 2);
    // printf("(C code) Sampled: [%.2lf %.2lf] result %lf \n", x, y, t);
    return t;
}

void learn(double *X_grid, bool *sampled, int *X, double *T, int t, double *mu, double *sigma,
                    double(*kernel)(double *, double *, double *, double *), double beta, int n) {
    /*
     * grid_idx = self.argmax_ucb()
    *  self.sample(self.X_grid[grid_idx])
    *  for every point x:
     *  gp_regression()
    *  gp.fit(self.X, self.T)
    *  mu1 = self.mu
     */
    int maxI = 0;
    int maxJ = 0;
    double max = mu[0] + sqrt(beta) * sigma[0];
    int inj = 0;
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double currentValue = mu[inj] + sqrt(beta) * sigma[inj];

            /*
            double x = X_grid[i * 2 * n + 2 * j];
            double y = X_grid[i * 2 * n + 2 * j + 1];
            if (debug && (x == 0 && y == -2.25 || x == 0 && y == -2)) {
                printf("[x, y] = [%.2lf, %.2lf], mu[xy]: %lf, sigma[xy]: %lf, cv: %lf, alreadySampled: %d\n", x, y,
                       mu[i * n + j], sigma[i * n + j],
                       currentValue, sampled[i * n + j]);
            }
             */
            if (currentValue > max && !sampled[inj] ) { // Maybe faster if inverted, especially after blocking? Lookup faster than comparison.
                max = currentValue;
                maxI = i;
                maxJ = j;
            }
            inj++;
        }
    }

    const int t2 = 2 * t;
    X[t2] = maxI;
    X[t2 + 1] = maxJ;
    const int maxInmaxJ = maxI * n + maxJ;
    sampled[maxInmaxJ] = true;
    const int maxInmaxJ2 = 2 * maxInmaxJ;
    T[t] = function(X_grid[maxInmaxJ2], X_grid[maxInmaxJ2 + 1]);
    gp_regression(X_grid, X, T, t, kernel, mu, sigma, n); // updating mu and sigma for every x in X_grid
}

double kernel2(double *x1, double *y1, double *x2, double *y2) {
    // RBF kernel
    double sigma = 1;
    const double x1_x2 = *x1 - *x2;
    const double y1_y2 = *y1 - *y2;
    const double x1212 = x1_x2 * x1_x2;
    const double y1212 = y1_y2 * y1_y2;
    const double numerator = x1212 + y1212;
    const double arg = numerator / 2;
    const double negArg = -arg;
    
    // return exp(-(x1212 + y1212) / (2 * sigma * sigma));
    return exp(negArg);
}

// Not timed... doesn't have to be optimized at the moment.
void initialize_meshgrid(double *X_grid, int n, double min, double inc) {
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

void gpucb_initialized(int maxIter,
                                int n,
                                double *T,
                                int *X,
                                double *X_grid,
                                bool *sampled,
                                double *mu,
                                double *sigma,
                                double beta) {
    for (int t = 0; t < maxIter; t++) {
        learn(X_grid, sampled, X, T, t, mu, sigma, kernel2, beta, n);
    }
}

int gpucb(int maxIter, int n, double grid_min, double grid_inc) {

    // Allocations
    double T[maxIter];
    int X[2 * maxIter];
    double X_grid[2 * n * n];
    bool sampled[n * n];
    double mu[n * n];
    double sigma[n * n];
    const double beta = 100;

    // Initializations
    for (int i = 0; i < n * n; i++) {
        sampled[i] = false;
        mu[i] = 0;
        sigma[i] = 0.5;
    }
    initialize_meshgrid(X_grid, n, grid_min, grid_inc);


    // -------------------------------------------------------------
    //                  Done with initializations
    // -------------------------------------------------------------

    gpucb_initialized(maxIter, n, T, X, X_grid, sampled, mu, sigma, beta);

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