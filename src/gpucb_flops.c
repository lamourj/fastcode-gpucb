// This version includes the incremental cholesky factorization.
#include "gpucb_flops.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>

const char *tag[10] = {"flops_c"};

float add(float a, float b) {
    _N_ADDS += 1;
    return a + b;
}

float sub(float a, float b) {
    _N_ADDS += 1;
    return a - b;
}

float divide(float a, float b) {
    _N_DIVS += 1;
    return a / b;
}

float fexponential(float a) {
    _N_EXPS += 1;
    return expf(a);
}

float mult(float a, float b) {
    _N_MULS += 1;
    return a * b;
}

float square_root(float a) {
    _N_SQRT += 1;
    return sqrtf(a);
}

void initialize(const int I, const int N) {
    printf("Init baseline\n");
    BETA_ = 100;
    GRID_MIN_ = -6;
    GRID_INC_ = 0.025;

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

    _N_ADDS = 0;
    _N_DIVS = 0;
    _N_EXPS = 0;
    _N_MULS = 0;
    _N_SQRT = 0;

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

void initialize_meshgrid_baseline(float *X_grid, int n, float min, float inc) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            X_grid[i * 2 * n + 2 * j] = min + j * inc;
            X_grid[i * 2 * n + 2 * j + 1] =
                    min + i * inc; // With this assignment, meshgrid is the same as python code
        }
    }
}

float function_baseline(float x, float y) {
    // float t = sin(x) + cos(y);
    // float t = -powf(x, 2) - powf(y, 2);
    float xx = mult(x, x);
    float yy = mult(y, y);
    float xx_neg = sub(0, xx);
    float t = sub(xx_neg, yy);
    return t;
}

void learn_baseline(float *X_grid,
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

    int maxI = 0;
    int maxJ = 0;
    float max = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float currentValue = add(mu[i * n + j], mult(sqrtf(beta), sigma[i * n + j]));

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
    gp_regression_baseline(X_grid, K, L_T, X, T, t, maxIter, kernel, mu, sigma,
                           n); // updating mu and sigma for every x in X_grid
}

float kernel2_baseline(float *x1, float *y1, float *x2, float *y2) {
    // RBF kernel
    float x1x2 = sub(*x1, *x2);
    float y1y2 = sub(*y1, *y2);
    float x1x2_squared = mult(x1x2, x1x2);
    float y1y2_squared = mult(y1y2, y1y2);
    float num = add(x1x2_squared, y1y2_squared);
    float operand = sub(0, divide(num, 2.f));
    float t = fexponential(operand);
    return t;
}

void run() {
    for (int t = 0; t < I_; t++) {
        learn_baseline(X_grid_, K_, L_, sampled_, X_, T_, t, I_, mu_, sigma_, kernel2_baseline, BETA_, N_);
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

    printf("N=%d, I=%d: add: %lu, mult: %lu, div: %lu, exp: %lu\n", N_, I_, _N_ADDS, _N_MULS, _N_DIVS, _N_EXPS);
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
void incremental_cholesky_baseline(float *A, float *A_T, int n1, int n2, int size) {
    for (int i = n1; i < n2; ++i) {
        // Update the off diagonal entries.
        for (int j = 0; j < i; ++j) {
            for (int k = 0; k < j; ++k) {

                A[size * i + j] = sub(A[size * i + j], mult(A[size * i + k], A[size * j + k]));
            }
            A[size * i + j] = divide(A[size * i + j], A[size * j + j]);
            A_T[size * j + i] = A[size * i + j];
        }
        // Update the diagonal entry.
        for (int k = 0; k < i; ++k) {
            A[size * i + i] = sub(A[size * i + i], mult(A[size * i + k], A[size * i + k]));
        }
        A[size * i + i] = square_root(A[size * i + i]);
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
void cholesky_solve2_baseline(int d, int size, float *LU, float *b, float *x, int lower) {
    if (lower == 1) {
        for (int i = 0; i < d; ++i) {
            float sum = 0;
            for (int k = 0; k < i; ++k) {

                sum = add(sum, mult(LU[i * size + k], x[k]));
            }
            x[i] = divide(sub(b[i], sum), LU[i * size + i]);
        }
    } else {
        for (int i = d - 1; i >= 0; --i) {
            float sum = 0;
            for (int k = i + 1; k < d; ++k) {
                sum = add(sum, mult(LU[i * size + k], x[k]));
            }
            x[i] = divide(sub(b[i], sum), LU[i * size + i]);
        }
    }

}


void gp_regression_baseline(float *X_grid,
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
            K[i * maxIter + j] = add(K[i * maxIter + j], 0.5);
            // K[i * maxIter + j] += 0.5;
        }
    }


    // 2. Cholesky
    incremental_cholesky_baseline(K, L_T, t_gp - 1, t_gp, maxIter);

    // 3. Compute alpha
    float x[t_gp];
    float alpha[t_gp];
    float v[t_gp];


    cholesky_solve2_baseline(t_gp, maxIter, K, T, x, 1);
    cholesky_solve2_baseline(t_gp, maxIter, L_T, x, alpha, 0);

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
                f_star = add(f_star, mult(k_star[k], alpha[k]));
            }

            mu[i * n + j] = f_star;
            cholesky_solve2_baseline(t_gp, maxIter, K, k_star, v, 1);

            float variance = 1.0;
            for (int k = 0; k < t_gp; k++) {
                variance = sub(variance, mult(v[k], v[k]));
            }


            if (variance < 0) {
                variance = 0.0;
            }
            sigma[i * n + j] = variance;

        }
    }
}
