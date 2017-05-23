// Baseline version without incremental cholesky.

#include "gpucb0.h"
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

double function_baseline(double x, double y) {
    // double t = sin(x) + cos(y);
    double t = -pow(x, 2) - pow(y, 2);
    printf("(C code) Sampled: [%.2lf %.2lf] result %lf \n", x, y, t);
    return t;
}

void learn_baseline(double *X_grid, bool *sampled, int *X, double *T, int t, double *mu, double *sigma,
                    double(*kernel)(double *, double *, double *, double *), double beta, int n) {
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
            /*
            double x = X_grid[i * 2 * n + 2 * j];
            double y = X_grid[i * 2 * n + 2 * j + 1];
            if (debug && (x == 0 && y == -2.25 || x == 0 && y == -2)) {
                printf("[x, y] = [%.2lf, %.2lf], mu[xy]: %lf, sigma[xy]: %lf, cv: %lf, alreadySampled: %d\n", x, y,
                       mu[i * n + j], sigma[i * n + j],
                       currentValue, sampled[i * n + j]);
            }
             */
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
    T[t] = function_baseline(X_grid[maxI * 2 * n + 2 * maxJ], X_grid[maxI * 2 * n + 2 * maxJ + 1]);
    gp_regression_baseline(X_grid, X, T, t, kernel, mu, sigma, n); // updating mu and sigma for every x in X_grid
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

void gpucb_initialized_baseline(int maxIter,
                                int n,
                                double *T,
                                int *X,
                                double *X_grid,
                                bool *sampled,
                                double *mu,
                                double *sigma,
                                double beta) {
    for (int t = 0; t < maxIter; t++) {
        learn_baseline(X_grid, sampled, X, T, t, mu, sigma, kernel2_baseline, beta, n);
    }
}

int gpucb_baseline(int maxIter, int n, double grid_min, double grid_inc) {

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
    initialize_meshgrid_baseline(X_grid, n, grid_min, grid_inc);


    // -------------------------------------------------------------
    //                  Done with initializations
    // -------------------------------------------------------------

    gpucb_initialized_baseline(maxIter, n, T, X, X_grid, sampled, mu, sigma, beta);

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

/*
 Straightforward implementation of inplace Cholesky decomposition of matrix A.
 Input arguments:
    A:    The matrix to decompose
    n:    The size of the data in matrix A to decompose
    size: The actual size of the rows
 */
void cholesky_baseline(double *A, int n, int size) {
    for (int i = 0; i < n; ++i) {

        // Update the off diagonal entries first.
        for (int j = 0; j < i; ++j) {
            for (int k = 0; k < j; ++k) {
                A[size * i + j] -= A[size * i + k] * A[size * j + k];
            }
            A[size * i + j] /= A[size * j + j];
        }

        // Update the diagonal entry of this row.
        for (int k = 0; k < i; ++k) {
            A[size * i + i] -= A[size * i + k] * A[size * i + k];
        }
        A[size * i + i] = sqrt(A[size * i + i]);
    }
}

/*
 Incremental implementation of Cholesky decomposition:
 The matrix contains a Cholesky decomposition until row n1,
 rows n1, to n2 are new data.
 Input arguments:
    A:    Partially decomposed matrix with new data from row n1, to n2
    n1:   Start of the new data
    n2:   End of the new data
    size: The actual size of the rows

 */
void incremental_cholesky_baseline(float *A, int n1, int n2, int size) {
    for (int i = n1; i < n2; ++i) {
        // Update the off diagonal entries.
        for (int j = 0; j < i; ++j) {
            for (int k = 0; k < j; ++k) {
                A[size * i + j] -= A[size * i + k] * A[size * j + k];
            }
            A[size * i + j] /= A[size * j + j];
        }
        // Update the diagonal entry.
        for (int k = 0; k < i; ++k) {
            A[size * i + i] -= A[size * i + k] * A[size * i + k];
        }
        A[size * i + i] = sqrtf(A[size * i + i]);
    }

}


/*
 * Solver for a matrix that is in Cholesky decomposition.
 * Input arguments:
 *      d: dimension of matrix
 *      LU: matrix
 *      b: right hand side
 *      x: vector to put result in
 *      lower: if one the lower triangle system is solved, else the upper triangle system is solved.
*/
void cholesky_solve2_baseline(int d, double *LU, double *b, double *x, int lower) {
    if (lower == 1) {
        for (int i = 0; i < d; ++i) {
            double sum = 0.;
            for (int k = 0; k < i; ++k) {
                sum += LU[i * d + k] * x[k];
            }
            x[i] = (b[i] - sum) / LU[i * d + i];
        }
    } else {
        for (int i = d - 1; i >= 0; --i) {
            double sum = 0.;
            for (int k = i + 1; k < d; ++k) {
                sum += LU[i * d + k] * x[k];
            }
            x[i] = (b[i] - sum) / LU[i * d + i];
        }
    }

}

// Old version.
void cholesky_solve_baseline(int d, double *LU, double *b, double *x) {
    double y[d];
    for (int i = 0; i < d; ++i) {
        double sum = 0.;
        for (int k = 0; k < i; ++k)sum += LU[i * d + k] * y[k];
        y[i] = (b[i] - sum) / LU[i * d + i];
    }
    for (int i = d - 1; i >= 0; --i) {
        double sum = 0.;
        for (int k = i + 1; k < d; ++k)sum += LU[k * d + i] * x[k];
        x[i] = (y[i] - sum) / LU[i * d + i];
    }
}


void transpose_baseline(double *M, double *M_T, int d) {
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            M_T[j * d + i] = M[i * d + j];
        }
    }
}


void gp_regression_baseline(double *X_grid,
                            int *X,
                            double *T,
                            int t,
                            double(*kernel)(double *, double *, double *, double *),
                            double *mu,
                            double *sigma,
                            int n) {
    int t_gp = t + 1;
    double L_T[t_gp * t_gp];
    double K[t_gp * t_gp];

    // Build the K matrix
    for (int i = 0; i < t_gp; i++) {
        for (int j = 0; j < t_gp; j++) {
            int x1 = X[2 * i];
            int y1 = X[2 * i + 1];
            int x2 = X[2 * j];
            int y2 = X[2 * j + 1];

            K[i * t_gp + j] = (*kernel)(&X_grid[x1 * 2 * n + 2 * y1], &X_grid[x1 * 2 * n + 2 * y1 + 1],
                                        &X_grid[x2 * 2 * n + 2 * y2], &X_grid[x2 * 2 * n + 2 * y2 + 1]);
            if(i == j){
                K[i * t_gp + j] += 0.5;
            }
            // K is symmetric, shouldn't go through all entries when optimizing

            /*printf("t_gp: %d, x0: %lf, y0: %lf, x1: %lf, y1: %lf, k: %lf, ki:%d \n", t_gp, X_grid[x1 * 2 * n + 2 * y1],
                   X_grid[x1 * 2 * n + 2 * y1 + 1], X_grid[x2 * 2 * n + 2 * y2], X_grid[x2 * 2 * n + 2 * y2 + 1],
                   K[i * t_gp + j], i * t_gp + j);*/

        }
        //printf("\n");
    }

    // 2. Cholesky
    cholesky_baseline(K, t_gp, t_gp);

    double *L = K;

    // 3. Compute alpha
    double x[t_gp];
    double alpha[t_gp];
    double v[t_gp];


    cholesky_solve2_baseline(t_gp, L, T, x, 1);

    transpose_baseline(L, L_T, t_gp); // TODO: Maybe do this more efficient
    cholesky_solve2_baseline(t_gp, L_T, x, alpha, 0);

    // 4-6. For all points in grid, compute k*, mu, sigma

    for (int i = 0; i < n; i++) // for all points in X_grid
    {
        for (int j = 0; j < n; j++) // for all points in X_grid
        {
            double x_star = X_grid[2 * n * i + 2 * j]; // Current grid point that we are looking at
            double y_star = X_grid[2 * n * i + 2 * j + 1];
            double k_star[t_gp];

            for (int k = 0; k < t_gp; k++) {
                int x = X[2 * k];
                int y = X[2 * k + 1];
                double arg1x = X_grid[x * 2 * n + 2 * y];
                double arg1y = X_grid[x * 2 * n + 2 * y + 1];
                k_star[k] = (*kernel)(&arg1x, &arg1y, &x_star, &y_star);
            }

            double f_star = 0;
            for (int k = 0; k < t_gp; k++) {
                //f_star += k_star[k] * alpha->data[k];
                f_star += k_star[k] * alpha[k];
                if(k_star[k] != k_star[k]) {
                    printf("k_star problem\n");
                }
                if(alpha[k] != alpha[k]) {
                    printf("alpha problem, k: %lf \n", k_star[k]);
                    return;
                }
            }

            mu[i * n + j] = f_star;
            if(f_star != f_star) {
                printf("Nan, %d\n", t);
            }
            //printf("fstar is: %lf", f_star);
            //printf("write in mu at %d \n", i*n+j);
            cholesky_solve2_baseline(t_gp, L, k_star, v, 1);
            //printf("loop solve done\n");

            double variance = (*kernel)(&x_star, &y_star, &x_star, &y_star);
            for (int k = 0; k < t_gp; k++) {
                //variance -= v->data[k] * v->data[k];
                variance -= v[k] * v[k];
            }

            if(variance < 0){
                variance = 0.0;
            }
            sigma[i * n + j] = variance;

        }
    }
}