// Inline the cholesky solve


#include "gpucb5.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

const char *tag[10] = {"inlined"};

void initialize(const int I, const int N) {
    if (N % 8 != 0) {
        printf("n is not divisible by 8 !!! \n");
    }

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

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        sampled_[i] = false;
        mu_[i] = 0;
        sigma_[i] = 0.5;
    }

    if (T_ == 0 || X_ == 0 || X_grid_ == 0 || sampled_ == 0 || mu_ == 0 || sigma_ == 0 || K_ == 0 || L_ == 0) {
        printf("ERROR: Out of memory\n");
    }

    initialize_meshgrid(X_grid_, N_, GRID_MIN_, GRID_INC_);
}

void initialize_meshgrid(float *X_grid, int n, float min, float inc) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            X_grid[i * 2 * n + 2 * j] = min + j * inc;
            X_grid[i * 2 * n + 2 * j + 1] =
                    min + i * inc; // With this assignment, meshgrid is the same as python code
        }
    }
}

float function(float x, float y) {
    // float t = sin(x) + cos(y);
    float t = -powf(x, 2) - powf(y, 2);
    printf("(C code) Sampled: [%.2f %.2f] result %f \n", x, y, t);
    return t;
}

void learn(float *X_grid,
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
    float max = mu[0] + sqrtf(beta) * sigma[0];
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
    T[t] = function(X_grid[maxI * 2 * n + 2 * maxJ], X_grid[maxI * 2 * n + 2 * maxJ + 1]);
    gp_regression_opt(X_grid, K, L_T, X, T, t, maxIter, kernel, mu, sigma,
                      n); // updating mu and sigma for every x in X_grid
}

float kernel2(float *x1, float *y1, float *x2, float *y2) {
    // RBF kernel
    float sigma = 1;
    return expf(-((*x1 - *x2) * (*x1 - *x2) + (*y1 - *y2) * (*y1 - *y2)) / (2 * sigma * sigma));
}

void run() {
    for (int t = 0; t < I_; t++) {
        learn(X_grid_, K_, L_, sampled_, X_, T_, t, I_, mu_, sigma_, kernel2, BETA_, N_);
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
void cholesky(float *A, int n, int size) {
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
void incremental_cholesky(float *A, float *A_T, int n1, int n2, int size) {
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
void cholesky_solve2(int d, int size, float *LU, float *b, float *x, int lower) {
    if (lower == 1) {
        for (int i = 0; i < d; ++i) {
            float sum = 0;
            for (int k = 0; k < i; ++k) {
                sum += LU[i * size + k] * x[k];
            }
            x[i] = (b[i] - sum) / LU[i * size + i];
        }
    } else {
        for (int i = d - 1; i >= 0; --i) {
            float sum = 0;
            for (int k = i + 1; k < d; ++k) {
                sum += LU[i * size + k] * x[k];
            }
            x[i] = (b[i] - sum) / LU[i * size + i];
        }
    }
}

void cholesky_solve2_opt(int d, int size, float *LU, float *b, float *x, int lower) {
    // TODO: Unroll over i ? Blocking (LU and x accessed several times)

    if (lower == 1) {
        float sum0 = 0;
        for (int i = 0; i < d; ++i) {
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;

            for (int k = 0; k + 3 < i; k += 4) {
                /*printf("k: %d\n", k);
                printf("k: %d\n", k+1);
                printf("k: %d\n", k+2);
                printf("k: %d\n", k+3);*/
                const int isizek = i * size + k;
                const float lu0 = LU[isizek];
                const float xk0 = x[k];

                const float lu1 = LU[isizek + 1];
                const float xk1 = x[k + 1];

                const float lu2 = LU[isizek + 2];
                const float xk2 = x[k + 2];

                const float lu3 = LU[isizek + 3];
                const float xk3 = x[k + 3];

                const float term0 = lu0 * xk0;
                const float term1 = lu1 * xk1;
                const float term2 = lu2 * xk2;
                const float term3 = lu3 * xk3;

                sum0 += term0;
                sum1 += term1;
                sum2 += term2;
                sum3 += term3;
            }
            const float bi = b[i];
            const float lu = LU[i * size + i];

            const float sum01 = sum0 + sum1;
            const float sum23 = sum2 + sum3;
            const float sum0123 = sum01 + sum23;

            float sumRest = 0;
            for (int k = 4 * (i / 4); k < i; k++) {
                // printf("k: %d\n", k);
                const float lu0 = LU[i * size + k];
                const float xk0 = x[k];
                const float term0 = lu0 * xk0;
                sumRest += term0;
            }

            const float sum = sum0123 + sumRest;
            const float num = bi - sum;
            const float xi = num / lu;
            x[i] = xi;
        }
    } else {
        for (int i = d - 1; i >= 0; --i) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;

            for (int k = i + 1; k + 3 < d; ++k) {
                const int isizek = i * size + k;

                const float lu0 = LU[isizek];
                const float xk0 = x[k];

                const float lu1 = LU[isizek + 1];
                const float xk1 = x[k + 1];

                const float lu2 = LU[isizek + 2];
                const float xk2 = x[k + 2];

                const float lu3 = LU[isizek + 3];
                const float xk3 = x[k + 3];

                const float term0 = lu0 * xk0;
                const float term1 = lu1 * xk1;
                const float term2 = lu2 * xk2;
                const float term3 = lu3 * xk3;

                sum0 += term0;
                sum1 += term1;
                sum2 += term2;
                sum3 += term3;
            }

            float sumRest = 0;
            const float sum01 = sum0 + sum1;
            const float sum23 = sum2 + sum3;
            const float sum0123 = sum01 + sum23;
            const float bi = b[i];
            const float lu = LU[i * size + i];


            for (int k = 4 * ((i + 1) / 4); k < d; k++) {
                printf("k: %d\n", k);
                const float lu0 = LU[i * size + k];
                const float xk0 = x[k];
                const float term0 = lu0 * xk0;
                sumRest += term0;
            }
            const float sum = sum0123 + sumRest;
            const float num = bi - sum;
            const float xi = num / lu;
            x[i] = xi;
        }
    }
}


void transpose(float *M, float *M_T, int d, int size) {
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            M_T[j * size + i] = M[i * size + j];
        }
    }
}


void gp_regression(float *X_grid,
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
    incremental_cholesky(K, L_T, t_gp - 1, t_gp, maxIter);

    // 3. Compute alpha
    float *x = (float *) malloc(t_gp * sizeof(float));
    float *alpha = (float *) malloc(t_gp * sizeof(float));
    float *v = (float *) malloc(t_gp * sizeof(float));


    cholesky_solve2(t_gp, maxIter, K, T, x, 1);
    cholesky_solve2(t_gp, maxIter, L_T, x, alpha, 0);

    // 4-6. For all points in grid, compute k*, mu, sigma

    for (i = 0; i < n; i++) { // for all points in X_grid ([i])
        for (int jj = 0; jj < n; jj += 8) { // for all points in X_grid ([i][j])
            // i, jj, kk, ll, j, k, l
            for (int j = jj; j < jj + 8; j++) {
                float x_star = X_grid[2 * n * i + 2 * j]; // Current grid point that we are looking at
                float y_star = X_grid[2 * n * i + 2 * j + 1];
                // float k_star[t_gp];
                float *k_star = (float *) malloc(t_gp * sizeof(float));
                float f_star = 0;
                float variance = (*kernel)(&x_star, &y_star, &x_star, &y_star);
                int x_, y_;
                float arg1x, arg1y, sum;

                for (int kk = 0; kk + 7 < t_gp; kk += 8) {
                    for (int k = kk; k < kk + 8; k++) {
                        x_ = X[2 * k];
                        y_ = X[2 * k + 1];
                        arg1x = X_grid[x_ * 2 * n + 2 * y_];
                        arg1y = X_grid[x_ * 2 * n + 2 * y_ + 1];
                        k_star[k] = (*kernel)(&arg1x, &arg1y, &x_star, &y_star);

                        sum = 0.0;
                        for (int ll = 0; ll + 7 < k; ll += 8) {
                            for (int l = ll; l < ll + 8; ++l) {
                                // This loop should be vectorized: Nice 8x8 block (MMM)
                                sum += K[k * maxIter + l] * v[l];
                            }
                        }
                        for (int l = 8 * (k / 8); l < k; ++l) {
                            // triangle
                            sum += K[k * maxIter + l] * v[l];
                        }

                        v[k] = (k_star[k] - sum) / K[k * maxIter + k];
                        f_star += k_star[k] * alpha[k];
                        variance -= v[k] * v[k];
                    }
                }
                free(k_star);
                for (int k = 8 * (t_gp / 8); k < t_gp; k++) {
                    x_ = X[2 * k];
                    y_ = X[2 * k + 1];
                    arg1x = X_grid[x_ * 2 * n + 2 * y_];
                    arg1y = X_grid[x_ * 2 * n + 2 * y_ + 1];
                    k_star[k] = (*kernel)(&arg1x, &arg1y, &x_star, &y_star);

                    sum = 0.0;
                    for (int ll = 0; ll + 7 < k; ll += 8) {
                        for (int l = ll; l < ll + 8; ++l) {
                            // l nice, k not nice
                            sum += K[k * maxIter + l] * v[l];
                        }
                    }
                    for (int l = 8 * (k / 8); l < k; ++l) {
                        // Triangle
                        sum += K[k * maxIter + l] * v[l];
                    }

                    v[k] = (k_star[k] - sum) / K[k * maxIter + k];
                    f_star += k_star[k] * alpha[k];
                    variance -= v[k] * v[k];
                }


                mu[i * n + j] = f_star;

                if (variance < 0) {
                    variance = 0.0;
                }
                sigma[i * n + j] = variance;
            }
        }
    }
    free(x);
    free(alpha);
    free(v);
}


void gp_regression_opt(float *X_grid,
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
    incremental_cholesky(K, L_T, t_gp - 1, t_gp, maxIter);

    // 3. Compute alpha
    // float x[t_gp];
    // float alpha[t_gp];
    // float v[t_gp];
    float *x = (float *) malloc(t_gp * sizeof(float));
    float *alpha = (float *) malloc(t_gp * sizeof(float));
    float *v = (float *) malloc(t_gp * sizeof(float));


    cholesky_solve2(t_gp, maxIter, K, T, x, 1);
    cholesky_solve2(t_gp, maxIter, L_T, x, alpha, 0);

    // 4-6. For all points in grid, compute k*, mu, sigma

    float *sums = (float *) malloc(8 * 8 * sizeof(float));
    float *k_star = (float *) malloc(t_gp * sizeof(float));
    for (i = 0; i < n; i++) { // for all points in X_grid ([i])
        for (int jj = 0; jj < n; jj += 8) { // for all points in X_grid ([i][j])
            for (int j = jj; j < jj + 8; j++) {
                mu[i * n + j] = 0;
                float x_star = X_grid[2 * n * i + 2 * j];
                float y_star = X_grid[2 * n * i + 2 * j + 1];
                sigma[i * n + j] = (*kernel)(&x_star, &y_star, &x_star, &y_star);
            }
            // float sums[8 * 8];
            for (int kk = 0; kk + 7 < t_gp; kk += 8) {
                for (int z = 0; z < 8 * 8; z++) {
                    sums[z] = 0;
                }
                for (int ll = 0; ll <= kk; ll += 8) {
                    for (int j = jj; j < jj + 8; j++) {
                        float x_star = X_grid[2 * n * i + 2 * j];
                        float y_star = X_grid[2 * n * i + 2 * j + 1];
                        // float k_star[t_gp];
                        int x_, y_;
                        float arg1x, arg1y;

                        for (int k = kk; k < kk + 8; k++) {
                            x_ = X[2 * k];
                            y_ = X[2 * k + 1];
                            arg1x = X_grid[x_ * 2 * n + 2 * y_];
                            arg1y = X_grid[x_ * 2 * n + 2 * y_ + 1];
                            k_star[k] = (*kernel)(&arg1x, &arg1y, &x_star, &y_star);
                            if (ll == kk) {
                                for (int l = ll; l < k; ++l) {
                                    sums[(k % 8) * 8 + j % 8] += K[k * maxIter + l] * v[l];
                                }
                                v[k] = (k_star[k] - sums[(k % 8) * 8 + j % 8]) / K[k * maxIter + k];
                                mu[i * n + j] += k_star[k] * alpha[k];
                                sigma[i * n + j] -= v[k] * v[k];
                            } else {
                                for (int l = ll; l < ll + 8; ++l) {
                                    sums[(k % 8) * 8 + j % 8] += K[k * maxIter + l] * v[l];
                                }
                            }
                        }
                    }
                }
            }

            for (int z = 0; z < 8 * 8; z++) {
                sums[z] = 0;
            }
            for (int k = 8 * (t_gp / 8); k < t_gp; k++) {
                for (int ll = 0; ll + 7 < k; ll += 8) {
                    for (int j = jj; j < jj + 8; j++) {
                        for (int l = ll; l < ll + 8; ++l) {
                            sums[(k % 8) * 8 + j % 8] += K[k * maxIter + l] * v[l];
                        }
                    }
                }
                for (int l = 8 * (k / 8); l < k; ++l) {
                    for (int j = jj; j < jj + 8; j++) {
                        sums[(k % 8) * 8 + j % 8] += K[k * maxIter + l] * v[l];
                    }
                }
                int x_, y_;
                float arg1x, arg1y;
                x_ = X[2 * k];
                y_ = X[2 * k + 1];
                arg1x = X_grid[x_ * 2 * n + 2 * y_];
                arg1y = X_grid[x_ * 2 * n + 2 * y_ + 1];
                for (int j = jj; j < jj + 8; j++) {
                    float x_star = X_grid[2 * n * i + 2 * j];
                    float y_star = X_grid[2 * n * i + 2 * j + 1];
                    float kstar = (*kernel)(&arg1x, &arg1y, &x_star, &y_star);
                    v[k] = (kstar - sums[(k % 8) * 8 + j % 8]) / K[k * maxIter + k];

                    mu[i * n + j] += kstar * alpha[k];
                    sigma[i * n + j] -= v[k] * v[k];
                }
            }
            for (int j = jj; j < jj + 8; j++) {
                if (sigma[i * n + j] < 0) {
                    sigma[i * n + j] = 0.0;
                }
            }
        }
    }

    free(k_star);
    free(sums);
    free(x);
    free(alpha);
    free(v);
}