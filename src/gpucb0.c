// Baseline version without incremental cholesky.

#include "gpucb0.h"
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <float.h>

const char *tag[30] = {"baseline-comp-chol"};

void initialize(const int I, const int N) {
    BETA_ = 100;
    GRID_MIN_ = -6;
    GRID_INC_ = 0.025;
    //tag = "baseline";
    //tag = (char*)malloc((strlen(tmp)+1) * sizeof(char));

    I_ = I;
    N_ = N;
    T_ = (float *) malloc(I * sizeof(float));
    X_ = (int *) malloc(2 * I * sizeof(int));
    X_grid_ = (float *) malloc(2 * N * N * sizeof(float));
    sampled_ = (bool *) malloc(N * N * sizeof(bool));
    mu_ = (float *) malloc(N * N * sizeof(float));
    sigma_ = (float *) malloc(N * N * sizeof(float));
    K_ = (float *) malloc(I * I * sizeof(float));
    L_ = (float *) malloc(I * I * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        sampled_[i] = false;
        mu_[i] = 0;
        sigma_[i] = 0.5;
    }

    if (T_ == 0 || X_ == 0 || X_grid_ == 0 || sampled_ == 0 || mu_ == 0 || sigma_ == 0 || K_ == 0 || L_ == 0) {
        printf("ERROR: Out of memory\n");
    }

    initialize_meshgrid_baseline(X_grid_, N_, GRID_MIN_, GRID_INC_);
}

float function_baseline(float x, float y) {
    // float t = sin(x) + cos(y);
    float t = -powf(x, 2) - powf(y, 2);
    printf("(C code) Sampled: [%.2f %.2f] result %f \n", x, y, t);
    return t;
}

void learn_baseline(float *X_grid, bool *sampled, int *X, float *T, int t, float *mu, float *sigma,
                    float(*kernel)(float *, float *, float *, float *), float beta, int n) {
    bool debug = true;
    int maxI = 0;
    int maxJ = 0;
    float max = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float currentValue = mu[i * n + j] + sqrtf(beta) * sigma[i * n + j];
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

float kernel2_baseline(float *x1, float *y1, float *x2, float *y2) {
    // RBF kernel
    float sigma = 1;
    return expf(-((*x1 - *x2) * (*x1 - *x2) + (*y1 - *y2) * (*y1 - *y2)) / (2 * sigma * sigma));
}

void initialize_meshgrid_baseline(float *X_grid, int n, float min, float inc) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            X_grid[i * 2 * n + 2 * j] = min + j * inc;
            X_grid[i * 2 * n + 2 * j + 1] =
                    min + i * inc; // With this assignment, meshgrid is the same as python code
        }
    }
}

void run() {
    for (int t = 0; t < I_; t++) {
        learn_baseline(X_grid_, sampled_, X_, T_, t, mu_, sigma_, kernel2_baseline, BETA_, N_);
    }
}

void clean() {
    free(T_);
    free(X_);
    free(X_grid_);
    free(sampled_);
    free(mu_);
    free(sigma_);
    free(K_);
    free(L_);

}

/*
 Straightforward implementation of inplace Cholesky decomposition of matrix A.
 Input arguments:
    A:    The matrix to decompose
    n:    The size of the data in matrix A to decompose
    size: The actual size of the rows
 */
void cholesky_baseline(float *A, float *A_T, int n, int size) {
    for (int i = 0; i < n; ++i) {

        // Update the off diagonal entries first.
        for (int j = 0; j < i; ++j) {
            for (int k = 0; k < j; ++k) {
                A[size * i + j] -= A[size * i + k] * A[size * j + k];
            }
            A[size * i + j] /= A[size * j + j];
            A_T[size * j + i] = A[size * i + j];
        }

        // Update the diagonal entry of this row.
        for (int k = 0; k < i; ++k) {
            A[size * i + i] -= A[size * i + k] * A[size * i + k];
        }
        A[size * i + i] = sqrtf(A[size * i + i]);
        A_T[size * i + i] = A[size * i + i];
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
        A[size * i + i] = sqrtff(A[size * i + i]);
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
void cholesky_solve2_baseline(int d, float *LU, float *b, float *x, int lower) {
    if (lower == 1) {
        for (int i = 0; i < d; ++i) {
            float sum = 0.;
            for (int k = 0; k < i; ++k) {
                sum += LU[i * d + k] * x[k];
            }
            x[i] = (b[i] - sum) / LU[i * d + i];
        }
    } else {
        for (int i = d - 1; i >= 0; --i) {
            float sum = 0.;
            for (int k = i + 1; k < d; ++k) {
                sum += LU[i * d + k] * x[k];
            }
            x[i] = (b[i] - sum) / LU[i * d + i];
        }
    }

}

// Old version.
void cholesky_solve_baseline(int d, float *LU, float *b, float *x) {
    float y[d];
    for (int i = 0; i < d; ++i) {
        float sum = 0.;
        for (int k = 0; k < i; ++k)sum += LU[i * d + k] * y[k];
        y[i] = (b[i] - sum) / LU[i * d + i];
    }
    for (int i = d - 1; i >= 0; --i) {
        float sum = 0.;
        for (int k = i + 1; k < d; ++k)sum += LU[k * d + i] * x[k];
        x[i] = (y[i] - sum) / LU[i * d + i];
    }
}


void transpose_baseline(float *M, float *M_T, int d) {
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            M_T[j * d + i] = M[i * d + j];
        }
    }
}


void gp_regression_baseline(float *X_grid,
                            int *X,
                            float *T,
                            int t,
                            float(*kernel)(float *, float *, float *, float *),
                            float *mu,
                            float *sigma,
                            int n) {
    int t_gp = t + 1;
    float L_T[t_gp * t_gp];
    float K[t_gp * t_gp];

    // Build the K matrix
    for (int i = 0; i < t_gp; i++) {
        for (int j = 0; j < t_gp; j++) {
            int x1 = X[2 * i];
            int y1 = X[2 * i + 1];
            int x2 = X[2 * j];
            int y2 = X[2 * j + 1];

            K[i * t_gp + j] = (*kernel)(&X_grid[x1 * 2 * n + 2 * y1], &X_grid[x1 * 2 * n + 2 * y1 + 1],
                                        &X_grid[x2 * 2 * n + 2 * y2], &X_grid[x2 * 2 * n + 2 * y2 + 1]);
            if (i == j) {
                K[i * t_gp + j] += 0.5;
            }
            // K is symmetric, shouldn't go through all entries when optimizing
        }
    }

    // 2. Cholesky
    cholesky_baseline(K, L_T, t_gp, t_gp);

    float *L = K;

    // 3. Compute alpha
    float x[t_gp];
    float alpha[t_gp];
    float v[t_gp];


    cholesky_solve2_baseline(t_gp, L, T, x, 1);

    //transpose_baseline(L, L_T, t_gp); // TODO: Maybe do this more efficient
    cholesky_solve2_baseline(t_gp, L_T, x, alpha, 0);

    // 4-6. For all points in grid, compute k*, mu, sigma

    for (int i = 0; i < n; i++) // for all points in X_grid
    {
        for (int j = 0; j < n; j++) // for all points in X_grid
        {
            float x_star = X_grid[2 * n * i + 2 * j]; // Current grid point that we are looking at
            float y_star = X_grid[2 * n * i + 2 * j + 1];
            float k_star[t_gp];

            for (int k = 0; k < t_gp; k++) {
                int x = X[2 * k];
                int y = X[2 * k + 1];
                float arg1x = X_grid[x * 2 * n + 2 * y];
                float arg1y = X_grid[x * 2 * n + 2 * y + 1];
                k_star[k] = (*kernel)(&arg1x, &arg1y, &x_star, &y_star);
            }

            float f_star = 0;
            for (int k = 0; k < t_gp; k++) {
                //f_star += k_star[k] * alpha->data[k];
                f_star += k_star[k] * alpha[k];
                if (k_star[k] != k_star[k]) {
                    printf("k_star problem\n");
                }
                if (alpha[k] != alpha[k]) {
                    printf("alpha problem, k: %f \n", k_star[k]);
                    return;
                }
            }

            mu[i * n + j] = f_star;
            if (f_star != f_star) {
                printf("Nan, %d\n", t);
            }
            cholesky_solve2_baseline(t_gp, L, k_star, v, 1);
            //printf("loop solve done\n");

            float variance = 1.0;
            for (int k = 0; k < t_gp; k++) {
                //variance -= v->data[k] * v->data[k];
                variance -= v[k] * v[k];
            }

            if (variance < 0) {
                variance = 0.0;
            }
            sigma[i * n + j] = variance;

        }
    }
}