#include "baseline.c"
//#include "math.h"


double function(double x, double y) {
    double t = sin(x) + cos(y);
    printf("sampled: [%.1lf %.1lf] result %lf \n", x, y, t);
    return t;
}

void learn(double *X_grid, int *X, double *T, int t, double *mu, double *sigma,
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
            if (currentValue > max) {
                max = currentValue;
                maxI = i;
                maxJ = j;
            }
        }
    }

    X[2 * t] = maxI;
    X[2 * t + 1] = maxJ;

    T[t] = function(X_grid[maxI * 2 * n + 2 * maxJ], X_grid[maxI * 2 * n + 2 * maxJ + 1]);
    gp_regression(X_grid, X, T, t, kernel, mu, sigma, n); // updating mu and sigma for every x in X_grid
}

double kernel2(double *x1, double *y1, double *x2, double *y2) {
    // RBF kernel
    double sigma = 1;
    return exp((-(*x1 - *x2) * (*x1 - *x2) + (*y1 - *y2) * (*y1 - *y2)) / (2 * sigma * sigma));
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

int main() {
    printf("Welcome\n");
    // Define D, mu_0, sigma_0, kernel function k

    int maxIter = 1;
    double T[maxIter];
    int X[2 * maxIter];

    int n;
    n = 3;
    double X_grid[2 * n * n];
    double mu[n * n];
    double sigma[n * n];
    for (int i = 0; i < n * n; i++) {
        mu[i] = 0;
        sigma[i] = 0.5;
    }

    double grid_min = 0;
    double grid_inc = 1;
    initialize_meshgrid(X_grid, n, grid_min, grid_inc);

    double beta;
    beta = 100;

    for (int t = 0; t < maxIter; t++) {
        learn(X_grid, X, T, t, mu, sigma, kernel2, beta, n);
    }

    printf("Mu matrix after training: \n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.5lf ", mu[i * n + j]);
        }
        printf("\n");
    }

    return 0;
}