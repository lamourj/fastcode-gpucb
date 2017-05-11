#include "baseline.c"
#include "time.h"
#include <stdbool.h>

double function(double x, double y) {
    //double t = sin(x) + cos(y);
    double t = -pow(x, 2) - pow(y, 2);
    // printf("sampled: [%.2lf %.2lf] result %lf \n", x, y, t);
    return t;
}

void learn(double *X_grid, bool* sampled, int *X, double *T, int t, double *mu, double *sigma,
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
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double currentValue = mu[i * n + j] + beta * sigma[i * n + j];
            if (currentValue > max && !sampled[i * n + j]) {
                max = currentValue;
                maxI = i;
                maxJ = j;
            }
        }
    }

    X[2 * t] = maxI;
    X[2 * t + 1] = maxJ;
    sampled[maxI * n + maxJ] = true;
    T[t] = function(X_grid[maxI * 2 * n + 2 * maxJ], X_grid[maxI * 2 * n + 2 * maxJ + 1]);
    gp_regression(X_grid, X, T, t, kernel, mu, sigma, n); // updating mu and sigma for every x in X_grid
}

double kernel2(double *x1, double *y1, double *x2, double *y2) {
    // RBF kernel
    double sigma = 1;
    return exp(-((*x1 - *x2) * (*x1 - *x2) + (*y1 - *y2) * (*y1 - *y2)) / (2 * sigma * sigma));
}

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

int gpucb(int maxIter, int n, double grid_min, double grid_inc) {
    // printf("Welcome\n");
    // Define D, mu_0, sigma_0, kernel function k

    double T[maxIter];
    int X[2 * maxIter];

    double X_grid[2 * n * n];
    bool sampled[n * n];
    for(int i = 0; i < n * n; i++){
        sampled[i] = false;
    }
    double mu[n * n];
    double sigma[n * n];
    for (int i = 0; i < n * n; i++) {
        mu[i] = 0;
        sigma[i] = 0.5;
    }

    initialize_meshgrid(X_grid, n, grid_min, grid_inc);

    double beta;
    beta = 100;

    for (int t = 0; t < maxIter; t++) {
        learn(X_grid, sampled, X, T, t, mu, sigma, kernel2, beta, n);
    }


    FILE *f = fopen("mu_c.txt", "w");
    printf("Mu matrix after training: \n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(f, "%lf, ", mu[i * n + j]);
            //printf("%.5lf ", mu[i * n + j]);

        }
        fprintf(f, "\n");
    }

    printf("Sigma matrix after training: \n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            //printf("%.5lf ", sigma[i * n + j]);
        }
        printf("\n");
    }
    fclose(f);

    return 0;
}

int main() {
    int n, i, k;
    const int nIter = 20;
    const int nMin = 20;
    const int nMax = 5000;
    const int nInc = 20;
    const int nRuns = 300;
    const double cycles_per_second = 2.5 * pow(10, 9);
    const double grid_min = -3;
    const double grid_inc = 0.25;

    gpucb(2, 24, grid_min, grid_inc);
//    double seconds_per_it[(int) ceil((nMax - nMin) / nInc)];
//
//    k = 0;
//    for (n = nMin; n <= nMax; n+=nInc) {
//        for(i = 0; i < 10; i++){
//            gpucb(nIter, n, grid_min, grid_inc);
//        }
//        clock_t start = clock();
//        for (i = 0; i < nRuns; i++) {
//            gpucb(nIter, n, grid_min, grid_inc);
//        }
//        clock_t end = clock();
//        double time_elapsed_in_seconds = (end - start) / (double) CLOCKS_PER_SEC;
//        double seconds_per_iteration = time_elapsed_in_seconds / nRuns;
//        double cycles_per_iteration = seconds_per_iteration * cycles_per_second;
//        printf("n: %d, seconds/it: %lf, cycles/it: %lf\n", n, seconds_per_iteration, cycles_per_iteration);
//        seconds_per_it[k++] = seconds_per_iteration;
//    }
//
//    FILE *f = fopen("runtimes.txt", "w");
//    fprintf(f, "n, seconds_per_it, cycles_per_it \n");
//    for(k = 0; k < (int) ceil((nMax - nMin) / nInc); k++) {
//        fprintf(f, "%d, %d, %lf, %lf \n", k * nInc + nMin, nIter, seconds_per_it[k], seconds_per_it[k] * cycles_per_second);
//    }
//    fclose(f);
    return 0;
}