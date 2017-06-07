// This version includes the incremental cholesky factorization.
#include "baseline.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>

void initialize_bl(const int I, const int N) {
    BETA_bl = 100;
    GRID_MIN_bl = -6;
    GRID_INC_bl = 0.025;
    //tag = "baseline";
    //tag = (char*)malloc((strlen(tmp)+1) * sizeof(char));

    I_bl = I;
    N_bl = N;
    T_bl = (float *) malloc(I * sizeof(float));
    X_bl = (int *) malloc(2 * I * sizeof(int));
    X_grid_bl = (float *) malloc(2 * N * N * sizeof(float));
    sampled_bl = (bool *) malloc(N * N * sizeof(bool));
    mu_bl = (float *) malloc(N * N * sizeof(float));
    sigma_bl = (float *) malloc(N * N * sizeof(float));
    K_bl = (float *) malloc(I * I * sizeof(float));
    L_bl = (float *) malloc(I * I * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        sampled_bl[i] = false;
        mu_bl[i] = 0;
        sigma_bl[i] = 0.5;
    }

    if (T_bl == 0 || X_bl == 0 || X_grid_bl == 0 || sampled_bl == 0 || mu_bl == 0 || sigma_bl == 0 || K_bl == 0 ||
        L_bl == 0) {
        printf("ERROR: Out of memory\n");
    }

    initialize_meshgrid_bl(X_grid_bl, N_bl, GRID_MIN_bl, GRID_INC_bl);
}

void initialize_meshgrid_bl(float *X_grid, int n, float min, float inc) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            X_grid[i * 2 * n + 2 * j] = min + j * inc;
            X_grid[i * 2 * n + 2 * j + 1] =
                    min + i * inc; // With this assignment, meshgrid is the same as python code
        }
    }
}

float function_bl(float x, float y) {
    // float t = sin(x) + cos(y);
    float t = -powf(x, 2) - powf(y, 2);
    //printf("(baseline) Sampled: [%.2f %.2f] result %f \n", x, y, t);
    return t;
}

void learn_bl(float *X_grid,
              float *K,
              float *L_T,
              bool *sampled,
              int *X,
              float *T,
              int t,
              int maxIter,
              float *mu,
              float *sigma,
              float(*kernel)(float *, float *, float *, float *),
              const float beta,
              int n) {

    bool debug = true;
    int maxI = 0;
    int maxJ = 0;
    float max = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float currentValue = mu[i * n + j] + sqrtf(beta) * sigma[i * n + j];

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
    T[t] = function_bl(X_grid[maxI * 2 * n + 2 * maxJ], X_grid[maxI * 2 * n + 2 * maxJ + 1]);
    gp_regression_bl(X_grid, K, L_T, X, T, t, maxIter, kernel, mu, sigma,
                     n); // updating mu and sigma for every x in X_grid
}

float kernel2_bl(float *x1, float *y1, float *x2, float *y2) {
    // RBF kernel
    float sigma = 1;
    return expf(-((*x1 - *x2) * (*x1 - *x2) + (*y1 - *y2) * (*y1 - *y2)) / (2 * sigma * sigma));
}

void run_bl() {
    for (int t = 0; t < I_bl; t++) {
        learn_bl(X_grid_bl, K_bl, L_bl, sampled_bl, X_bl, T_bl, t, I_bl, mu_bl, sigma_bl, kernel2_bl, BETA_bl, N_bl);
    }
}

void clean_bl() {
    free(T_bl);
    free(X_bl);
    free(X_grid_bl);
    free(sampled_bl);
    free(mu_bl);
    free(sigma_bl);
    free(K_bl);
    free(L_bl);

}


/*
 Straightforward implementation of inplace Cholesky decomposition of matrix A.
 Input arguments:
    A:    The matrix to decompose
    n:    The size of the data in matrix A to decompose
    size: The actual size of the rows
 */
void cholesky_bl(float *A, int n, int size) {
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
        A[size * i + i] = sqrtf(A[size * i + i]);
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
void incremental_cholesky_bl(float *A, float *A_T, int n1, int n2, int size) {
    for (int i = n1; i < n2; ++i) {
        // Update the off diagonal entries.
        for (int j = 0; j < i; ++j) {
            for (int k = 0; k < j; ++k) {
                A[size * i + j] -= A[size * i + k] * A[size * j + k];
            }
            A[size * i + j] /= A[size * j + j];
            A_T[size * j + i] = A[size * i + j];
        }
        // Update the diagonal entry.
        for (int k = 0; k < i; ++k) {
            A[size * i + i] -= A[size * i + k] * A[size * i + k];
        }
        A[size * i + i] = sqrtf(A[size * i + i]);
        A_T[size * i + i] = A[size * i + i];
    }
}


/*
 * Solver for a matrix that is in Cholesky decomposition.
 * Input arguments:
 *      d: dimension of matrix
 *      size: the actual size of the matrix
 *      LU: matrix
 *      b: right hand side
 *      x: vector to put result in
 *      lower: if one the lower triangle system is solved, else the upper triangle system is solved.
*/
void cholesky_solve_bl(int d, int size, float *LU, float *b, float *x, int lower) {
    if (lower == 1) {
        for (int i = 0; i < d; ++i) {
            float sum = 0.;
            for (int k = 0; k < i; ++k) {
                sum += LU[i * size + k] * x[k];
            }
            x[i] = (b[i] - sum) / LU[i * size + i];
        }
    } else {
        for (int i = d - 1; i >= 0; --i) {
            float sum = 0.;
            for (int k = i + 1; k < d; ++k) {
                sum += LU[i * size + k] * x[k];
            }
            x[i] = (b[i] - sum) / LU[i * size + i];
        }
    }

}

void gp_regression_bl(float *X_grid,
                      float *K,
                      float *L_T,
                      int *X,
                      float *T,
                      int t,
                      int maxIter,
                      float   (*kernel)(float *, float *, float *, float *),
                      float *mu,
                      float *sigma,
                      int n) {
    int t_gp = t + 1;

    // extend the K matrix
    int i = t_gp - 1;
    for (int j = 0; j < t_gp; j++) {
        int x1 = X[2 * i];
        int y1 = X[2 * i + 1];
        int x2 = X[2 * j];
        int y2 = X[2 * j + 1];

        K[i * maxIter + j] = (*kernel)(&X_grid[x1 * 2 * n + 2 * y1], &X_grid[x1 * 2 * n + 2 * y1 + 1],
                                       &X_grid[x2 * 2 * n + 2 * y2], &X_grid[x2 * 2 * n + 2 * y2 + 1]);
        // K is symmetric, shouldn't go through all entries when optimizing
        if (i == j) {
            K[i * maxIter + j] += 0.5;
        }
    }


    // 2. Cholesky
    incremental_cholesky_bl(K, L_T, t_gp - 1, t_gp, maxIter);

    // 3. Compute alpha
    float x[t_gp];
    float alpha[t_gp];
    float v[t_gp];


    cholesky_solve_bl(t_gp, maxIter, K, T, x, 1);
    cholesky_solve_bl(t_gp, maxIter, L_T, x, alpha, 0);

    // 4-6. For all points in grid, compute k*, mu, sigma

    for (i = 0; i < n; i++) // for all points in X_grid ([i])
    {
        for (int j = 0; j < n; j++) // for all points in X_grid ([i][j])
        {
            float x_star = X_grid[2 * n * i + 2 * j]; // Current grid point that we are looking at
            float y_star = X_grid[2 * n * i + 2 * j + 1];
            float k_star[t_gp];

            for (int k = 0; k < t_gp; k++) {
                int x_ = X[2 * k];
                int y_ = X[2 * k + 1];
                float arg1x = X_grid[x_ * 2 * n + 2 * y_];
                float arg1y = X_grid[x_ * 2 * n + 2 * y_ + 1];
                k_star[k] = (*kernel)(&arg1x, &arg1y, &x_star, &y_star);
            }

            float f_star = 0;
            for (int k = 0; k < t_gp; k++) {
                //f_star += k_star[k] * alpha->data[k];
                f_star += k_star[k] * alpha[k];
            }

            mu[i * n + j] = f_star;
            cholesky_solve_bl(t_gp, maxIter, K, k_star, v, 1);

            float variance = 1.0;
            for (int k = 0; k < t_gp; k++) {
                variance -= v[k] * v[k];
            }


            if (variance < 0) {
                variance = 0.0;
            }
            sigma[i * n + j] = variance;

        }
    }
}
