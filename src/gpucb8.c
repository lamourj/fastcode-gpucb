// Find maximum value (point to sample in next it.) while doing GP-regression update.


#include "gpucb8.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "avx_mathfun.h"

const char *tag[10] = {"gpcub8"};

void initialize(const int I, const int N) {
    printf("Init gpucb8\n");
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
    maxIJ_ = malloc(2 * sizeof(int));
    maxIJ_[0] = 0;
    maxIJ_[1] = 0;

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
           int n,
           int *maxIJ) {

    int maxI = maxIJ[0];
    int maxJ = maxIJ[1];
    X[2 * t] = maxI;
    X[2 * t + 1] = maxJ;
    sampled[maxI * n + maxJ] = true;
    T[t] = function(X_grid[maxI * 2 * n + 2 * maxJ], X_grid[maxI * 2 * n + 2 * maxJ + 1]);
    gp_regression_opt(X_grid, K, L_T, X, T, t, maxIter, kernel, mu, sigma, sampled, beta,
                      n, maxIJ); // updating mu and sigma for every x in X_grid
}

float kernel2(float *x1, float *y1, float *x2, float *y2) {
    // RBF kernel
    float sigma = 1;
    return expf(-((*x1 - *x2) * (*x1 - *x2) + (*y1 - *y2) * (*y1 - *y2)) / (2 * sigma * sigma));
}

void run() {
    for (int t = 0; t < I_; t++) {
        learn(X_grid_, K_, L_, sampled_, X_, T_, t, I_, mu_, sigma_, kernel2, BETA_, N_, maxIJ_);
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


void solve_triangle(float *X_grid, int *X, float *mu, float *sigma, float *alpha, int i, int jj, int kk, int ll, int n,
                    int maxIter, float *sums, float *K, float *v) {
    for (int j = jj; j < jj + 8; j++) {
        float muinj = mu[i * n + j];
        float sigmainj = sigma[i * n + j];
        float x_star = X_grid[2 * n * i + 2 * j];
        float y_star = X_grid[2 * n * i + 2 * j + 1];
        int x_, y_;
        float arg1x, arg1y;

        for (int k = kk; k < kk + 8; k++) {
            x_ = X[2 * k];
            y_ = X[2 * k + 1];
            arg1x = X_grid[x_ * 2 * n + 2 * y_];
            arg1y = X_grid[x_ * 2 * n + 2 * y_ + 1];
            float kstar = expf(
                    -((arg1x - x_star) * (arg1x - x_star) + (arg1y - y_star) * (arg1y - y_star)) / 2.f);
            for (int l = ll; l < k; ++l) {
                sums[(k % 8) * 8 + j % 8] += K[k * maxIter + l] * v[l * 8 + (j % 8)];
            }
            v[k * 8 + (j % 8)] = (kstar - sums[(k % 8) * 8 + j % 8]) / K[k * maxIter + k];
            muinj += kstar * alpha[k];
            sigmainj -= v[k * 8 + (j % 8)] * v[k * 8 + (j % 8)];

        }
        mu[i * n + j] = muinj;
        sigma[i * n + j] = sigmainj;
    }
}


void solve_triangle_vect(float *X_grid, int *X, float *mu, float *sigma, float *alpha, int i, int jj, int kk, int ll,
                         int n, int maxIter, int k_max, float *sums, float *K, float *v) {
    __m256 mu_vector = _mm256_loadu_ps(mu + i * n + jj);
    __m256 sigma_vector = _mm256_loadu_ps(sigma + i * n + jj);
    const __m256 xy_star0 = _mm256_loadu_ps(X_grid + 2 * n * i + 2 * jj);
    const __m256 xy_star1 = _mm256_loadu_ps(X_grid + 2 * n * i + 2 * (jj + 4));
    __m256 xstar_h1 = _mm256_permute2f128_ps(xy_star0, xy_star1, 32);
    __m256 xstar_h2 = _mm256_permute2f128_ps(xy_star0, xy_star1, 49);
    const __m256 xstar_vector = _mm256_shuffle_ps(xstar_h1, xstar_h2, 136);
    const __m256 ystar_vector = _mm256_shuffle_ps(xstar_h1, xstar_h2, 221);


    int x_, y_;
    float arg1x, arg1y;


    for (int k = kk; k < kk + k_max; k++) {
        __m256 sum_vector = _mm256_loadu_ps(sums + 8 * (k % 8));
        x_ = X[2 * k];
        y_ = X[2 * k + 1];
        arg1x = X_grid[x_ * 2 * n + 2 * y_];
        arg1y = X_grid[x_ * 2 * n + 2 * y_ + 1];
        const __m256 arg1x_vector = _mm256_set1_ps(arg1x);
        const __m256 arg1y_vector = _mm256_set1_ps(arg1y);
        const __m256 minus_two_vector = _mm256_set1_ps(-2.f);

        __m256 exponentx = _mm256_sub_ps(arg1x_vector, xstar_vector);
        exponentx = _mm256_mul_ps(exponentx, exponentx);
        __m256 exponenty = _mm256_sub_ps(arg1y_vector, ystar_vector);
        exponenty = _mm256_mul_ps(exponenty, exponenty);
        __m256 exponent = _mm256_div_ps(_mm256_add_ps(exponentx, exponenty), minus_two_vector);
        __m256 kstar_vector = exp256_ps(exponent);

        __m256 K_vector;
        __m256 v_vector;
        for (int l = ll; l < k; ++l) {
            K_vector = _mm256_set1_ps(K[k * maxIter + l]);
            v_vector = _mm256_loadu_ps(v + l * 8);
            sum_vector = _mm256_fmadd_ps(K_vector, v_vector, sum_vector);
        }

        _mm256_storeu_ps(sums + (k % 8) * 8, sum_vector);

        __m256 K_diag = _mm256_set1_ps(K[k * maxIter + k]);
        v_vector = _mm256_sub_ps(kstar_vector, sum_vector);
        v_vector = _mm256_div_ps(v_vector, K_diag);
        _mm256_storeu_ps(v + k * 8, v_vector);

        mu_vector = _mm256_fmadd_ps(kstar_vector, _mm256_set1_ps(alpha[k]), mu_vector);

        v_vector = _mm256_loadu_ps(v + k * 8 + (jj % 8));

        sigma_vector = _mm256_fnmadd_ps(v_vector, v_vector, sigma_vector);

    }

    _mm256_storeu_ps(mu + i * n + jj, mu_vector);
    _mm256_storeu_ps(sigma + i * n + jj, sigma_vector);
}


void mmm(int jj, int kk, int ll, int maxIter, int k_max, float *sums, float *K, float *v) {
    for (int j = jj; j < jj + 8; j++) {
        for (int k = kk; k < kk + k_max; k++) {
            float tmp_sum = 0;
            for (int l = ll; l < ll + 8; ++l) {
                tmp_sum += K[k * maxIter + l] * v[l * 8 + (j % 8)];
            }
            sums[(k % 8) * 8 + j % 8] += tmp_sum;
        }
    }
}


void mmm_vect(int jj, int kk, int ll, int maxIter, int k_max, float *sums, float *K, float *v) {
    const __m256 v_row_0 = _mm256_loadu_ps(v + ll * 8);
    const __m256 v_row_1 = _mm256_loadu_ps(v + (ll + 1) * 8);
    const __m256 v_row_2 = _mm256_loadu_ps(v + (ll + 2) * 8);
    const __m256 v_row_4 = _mm256_loadu_ps(v + (ll + 4) * 8);
    const __m256 v_row_5 = _mm256_loadu_ps(v + (ll + 5) * 8);
    const __m256 v_row_6 = _mm256_loadu_ps(v + (ll + 6) * 8);
    const __m256 v_row_7 = _mm256_loadu_ps(v + (ll + 7) * 8);
    const __m256 v_row_3 = _mm256_loadu_ps(v + (ll + 3) * 8);

    const int kk_k_max = kk + k_max;

    for (int k = kk; k + 3 < kk_k_max; k += 4) {
        // k = 0
        __m256 sum_0 = _mm256_loadu_ps(sums + (k % 8) * 8);
        const int k_maxIter_ll_0 = k * maxIter + ll;
        const __m256 k_element_0_0 = _mm256_set1_ps(K[k_maxIter_ll_0]);
        sum_0 = _mm256_fmadd_ps(k_element_0_0, v_row_0, sum_0);
        const int k_maxIter_ll_1 = (k + 1) * maxIter + ll;
        const __m256 k_element_1_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 1]);
        const int k_maxIter_ll_2 = (k + 2) * maxIter + ll;
        const int k_maxIter_ll_3 = (k + 3) * maxIter + ll;
        const __m256 k_element_0_1 = _mm256_set1_ps(K[k_maxIter_ll_1]);
        __m256 sum_1 = _mm256_loadu_ps(sums + ((k + 1) % 8) * 8);
        sum_1 = _mm256_fmadd_ps(k_element_0_1, v_row_0, sum_1);


        const __m256 k_element_2_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 2]);


        const __m256 k_element_1_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 1]);
        const __m256 k_element_2_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 2]);

        __m256 sum_2 = _mm256_loadu_ps(sums + ((k + 2) % 8) * 8);

        const __m256 k_element_0_2 = _mm256_set1_ps(K[k_maxIter_ll_2]);
        const __m256 k_element_1_2 = _mm256_set1_ps(K[k_maxIter_ll_2 + 1]);
        const __m256 k_element_2_2 = _mm256_set1_ps(K[k_maxIter_ll_2 + 2]);


        __m256 sum_3 = _mm256_loadu_ps(sums + ((k + 3) % 8) * 8);

        const __m256 k_element_0_3 = _mm256_set1_ps(K[k_maxIter_ll_3]);
        const __m256 k_element_1_3 = _mm256_set1_ps(K[k_maxIter_ll_3 + 1]);
        const __m256 k_element_2_3 = _mm256_set1_ps(K[k_maxIter_ll_3 + 2]);

        const __m256 k_element_3_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 3]);
        const __m256 k_element_3_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 3]);
        sum_2 = _mm256_fmadd_ps(k_element_0_2, v_row_0, sum_2);
        const __m256 k_element_3_2 = _mm256_set1_ps(K[k_maxIter_ll_2 + 3]);
        sum_3 = _mm256_fmadd_ps(k_element_0_3, v_row_0, sum_3);
        const __m256 k_element_3_3 = _mm256_set1_ps(K[k_maxIter_ll_3 + 3]);

        sum_0 = _mm256_fmadd_ps(k_element_1_0, v_row_1, sum_0);
        const __m256 k_element_4_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 4]);
        sum_1 = _mm256_fmadd_ps(k_element_1_1, v_row_1, sum_1);
        const __m256 k_element_4_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 4]);
        sum_2 = _mm256_fmadd_ps(k_element_1_2, v_row_1, sum_2);
        const __m256 k_element_4_2 = _mm256_set1_ps(K[k_maxIter_ll_2 + 4]);
        sum_3 = _mm256_fmadd_ps(k_element_1_3, v_row_1, sum_3);
        const __m256 k_element_4_3 = _mm256_set1_ps(K[k_maxIter_ll_3 + 4]);

        sum_0 = _mm256_fmadd_ps(k_element_2_0, v_row_2, sum_0);
        const __m256 k_element_5_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 5]);
        sum_1 = _mm256_fmadd_ps(k_element_2_1, v_row_2, sum_1);
        const __m256 k_element_5_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 5]);
        sum_2 = _mm256_fmadd_ps(k_element_2_2, v_row_2, sum_2);
        const __m256 k_element_5_2 = _mm256_set1_ps(K[k_maxIter_ll_2 + 5]);
        sum_3 = _mm256_fmadd_ps(k_element_2_3, v_row_2, sum_3);
        const __m256 k_element_5_3 = _mm256_set1_ps(K[k_maxIter_ll_3 + 5]);

        sum_0 = _mm256_fmadd_ps(k_element_3_0, v_row_3, sum_0);
        const __m256 k_element_6_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 6]);
        sum_1 = _mm256_fmadd_ps(k_element_3_1, v_row_3, sum_1);
        const __m256 k_element_6_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 6]);
        sum_2 = _mm256_fmadd_ps(k_element_3_2, v_row_3, sum_2);
        const __m256 k_element_6_2 = _mm256_set1_ps(K[k_maxIter_ll_2 + 6]);
        sum_3 = _mm256_fmadd_ps(k_element_3_3, v_row_3, sum_3);
        const __m256 k_element_6_3 = _mm256_set1_ps(K[k_maxIter_ll_3 + 6]);

        sum_0 = _mm256_fmadd_ps(k_element_4_0, v_row_4, sum_0);
        const __m256 k_element_7_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 7]);
        sum_1 = _mm256_fmadd_ps(k_element_4_1, v_row_4, sum_1);
        const __m256 k_element_7_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 7]);
        sum_2 = _mm256_fmadd_ps(k_element_4_2, v_row_4, sum_2);
        const __m256 k_element_7_2 = _mm256_set1_ps(K[k_maxIter_ll_2 + 7]);
        sum_3 = _mm256_fmadd_ps(k_element_4_3, v_row_4, sum_3);
        const __m256 k_element_7_3 = _mm256_set1_ps(K[k_maxIter_ll_3 + 7]);

        sum_0 = _mm256_fmadd_ps(k_element_5_0, v_row_5, sum_0);
        sum_1 = _mm256_fmadd_ps(k_element_5_1, v_row_5, sum_1);
        sum_2 = _mm256_fmadd_ps(k_element_5_2, v_row_5, sum_2);
        sum_3 = _mm256_fmadd_ps(k_element_5_3, v_row_5, sum_3);
        sum_0 = _mm256_fmadd_ps(k_element_6_0, v_row_6, sum_0);
        sum_0 = _mm256_fmadd_ps(k_element_7_0, v_row_7, sum_0);
        _mm256_storeu_ps(sums + (k % 8) * 8, sum_0);

        sum_1 = _mm256_fmadd_ps(k_element_6_1, v_row_6, sum_1);
        sum_1 = _mm256_fmadd_ps(k_element_7_1, v_row_7, sum_1);
        _mm256_storeu_ps(sums + ((k + 1) % 8) * 8, sum_1);
        sum_2 = _mm256_fmadd_ps(k_element_6_2, v_row_6, sum_2);
        sum_2 = _mm256_fmadd_ps(k_element_7_2, v_row_7, sum_2);
        _mm256_storeu_ps(sums + ((k + 2) % 8) * 8, sum_2);
        sum_3 = _mm256_fmadd_ps(k_element_6_3, v_row_6, sum_3);
        sum_3 = _mm256_fmadd_ps(k_element_7_3, v_row_7, sum_3);

        _mm256_storeu_ps(sums + ((k + 3) % 8) * 8, sum_3);

    }
    for (int k = 4 * ((kk + k_max) / 4); k + 1 < kk_k_max; k += 2) {
        __m256 sum_0 = _mm256_loadu_ps(sums + (k % 8) * 8);
        __m256 sum_1 = _mm256_loadu_ps(sums + ((k + 1) % 8) * 8);
        const int k_maxIter_ll_0 = k * maxIter + ll;
        const __m256 k_element_0_0 = _mm256_set1_ps(K[k_maxIter_ll_0]);
        sum_0 = _mm256_fmadd_ps(k_element_0_0, v_row_0, sum_0);
        const int k_maxIter_ll_1 = (k + 1) * maxIter + ll;
        const __m256 k_element_1_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 1]);


        const __m256 k_element_2_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 2]);
        const __m256 k_element_0_1 = _mm256_set1_ps(K[k_maxIter_ll_1]);
        sum_0 = _mm256_fmadd_ps(k_element_1_0, v_row_1, sum_0);

        const __m256 k_element_1_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 1]);
        sum_0 = _mm256_fmadd_ps(k_element_2_0, v_row_2, sum_0);
        const __m256 k_element_2_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 2]);
        sum_1 = _mm256_fmadd_ps(k_element_0_1, v_row_0, sum_1);
        const __m256 k_element_3_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 3]);

        sum_1 = _mm256_fmadd_ps(k_element_1_1, v_row_1, sum_1);
        const __m256 k_element_4_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 4]);

        sum_1 = _mm256_fmadd_ps(k_element_2_1, v_row_2, sum_1);
        const __m256 k_element_5_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 5]);

        sum_1 = _mm256_fmadd_ps(k_element_3_1, v_row_3, sum_1);
        sum_1 = _mm256_fmadd_ps(k_element_5_1, v_row_5, sum_1);

        const __m256 k_element_6_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 6]);

        sum_1 = _mm256_fmadd_ps(k_element_6_1, v_row_6, sum_1);
        const __m256 k_element_3_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 3]);
        sum_0 = _mm256_fmadd_ps(k_element_3_0, v_row_3, sum_0);

        const __m256 k_element_4_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 4]);
        sum_0 = _mm256_fmadd_ps(k_element_4_0, v_row_4, sum_0);
        const __m256 k_element_5_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 5]);
        sum_0 = _mm256_fmadd_ps(k_element_5_0, v_row_5, sum_0);

        const __m256 k_element_6_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 6]);

        sum_0 = _mm256_fmadd_ps(k_element_6_0, v_row_6, sum_0);
        const __m256 k_element_7_0 = _mm256_set1_ps(K[k_maxIter_ll_0 + 7]);
        const __m256 k_element_7_1 = _mm256_set1_ps(K[k_maxIter_ll_1 + 7]);

        sum_0 = _mm256_fmadd_ps(k_element_7_0, v_row_7, sum_0);
        _mm256_storeu_ps(sums + (k % 8) * 8, sum_0);
        sum_1 = _mm256_fmadd_ps(k_element_4_1, v_row_4, sum_1);

        sum_1 = _mm256_fmadd_ps(k_element_7_1, v_row_7, sum_1);

        _mm256_storeu_ps(sums + ((k + 1) % 8) * 8, sum_1);
    }
    for (int k = 2 * ((kk + k_max) / 2); k < kk_k_max; k += 1) {
        const int k_maxIter_ll_0 = k * maxIter + ll;

        __m256 sum_0 = _mm256_loadu_ps(sums + (k % 8) * 8);

        const __m256 k_element_0 = _mm256_set1_ps(K[k_maxIter_ll_0]);
        const __m256 k_element_1 = _mm256_set1_ps(K[k_maxIter_ll_0 + 1]);
        const __m256 k_element_2 = _mm256_set1_ps(K[k_maxIter_ll_0 + 2]);

        sum_0 = _mm256_fmadd_ps(k_element_0, v_row_0, sum_0);
        const __m256 k_element_3 = _mm256_set1_ps(K[k_maxIter_ll_0 + 3]);

        sum_0 = _mm256_fmadd_ps(k_element_1, v_row_1, sum_0);
        const __m256 k_element_4 = _mm256_set1_ps(K[k_maxIter_ll_0 + 4]);

        sum_0 = _mm256_fmadd_ps(k_element_2, v_row_2, sum_0);
        const __m256 k_element_5 = _mm256_set1_ps(K[k_maxIter_ll_0 + 5]);

        sum_0 = _mm256_fmadd_ps(k_element_3, v_row_3, sum_0);
        const __m256 k_element_6 = _mm256_set1_ps(K[k_maxIter_ll_0 + 6]);

        sum_0 = _mm256_fmadd_ps(k_element_4, v_row_4, sum_0);
        const __m256 k_element_7 = _mm256_set1_ps(K[k_maxIter_ll_0 + 7]);

        sum_0 = _mm256_fmadd_ps(k_element_5, v_row_5, sum_0);

        sum_0 = _mm256_fmadd_ps(k_element_6, v_row_6, sum_0);

        sum_0 = _mm256_fmadd_ps(k_element_7, v_row_7, sum_0);

        _mm256_storeu_ps(sums + (k % 8) * 8, sum_0);
    }
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
                       bool *sampled,
                       float beta,
                       int n,
                       int *maxIJ) {
    int t_gp = t + 1;

    // extend the K matrix
    int i = t_gp - 1;
    for (int j = 0; j < t_gp; j++) {
        int x1 = X[2 * i];
        int y1 = X[2 * i + 1];
        int x2 = X[2 * j];
        int y2 = X[2 * j + 1];

        float _x1, _y1, _x2, _y2;
        _x1 = X_grid[x1 * 2 * n + 2 * y1];
        _y1 = X_grid[x1 * 2 * n + 2 * y1 + 1];
        _x2 = X_grid[x2 * 2 * n + 2 * y2];
        _y2 = X_grid[x2 * 2 * n + 2 * y2 + 1];
        float k_value = expf(-((_x1 - _x2) * (_x1 - _x2) + (_y1 - _y2) * (_y1 - _y2)) / 2.f);
        K[i * maxIter + j] = k_value;
        if (i == j) {
            K[i * maxIter + j] += 0.5;
        }
    }


    // 2. Cholesky
    incremental_cholesky(K, L_T, t_gp - 1, t_gp, maxIter);

    float *x = (float *) malloc(t_gp * sizeof(float));
    float *alpha = (float *) malloc(t_gp * sizeof(float));
    float *v = (float *) malloc(8 * t_gp * sizeof(float));


    cholesky_solve2(t_gp, maxIter, K, T, x, 1);
    cholesky_solve2(t_gp, maxIter, L_T, x, alpha, 0);

    // 4-6. For all points in grid, compute k*, mu, sigma


    float maxValue = -FLT_MAX;
    int maxI = 0, maxJ = 0;

    float *sums = (float *) malloc(8 * 8 * sizeof(float));
    for (i = 0; i < n; i++) { // for all points in X_grid ([i])
        for (int jj = 0; jj < n; jj += 8) { // for all points in X_grid ([i][j])
            for (int j = jj; j < jj + 8; j++) {
                mu[i * n + j] = 0;
                sigma[i * n + j] = 1.0;
            }
            for (int zz = 0; zz < 8 * t_gp; ++zz) {
                v[zz] = 0;
            }
            for (int kk = 0; kk + 7 < t_gp; kk += 8) {
                for (int z = 0; z < 8 * 8; z++) {
                    sums[z] = 0;
                }
                for (int ll = 0; ll <= kk; ll += 8) {
                    if (ll == kk) {
                        solve_triangle_vect(X_grid, X, mu, sigma, alpha, i, jj, kk, ll, n, maxIter, 8, sums, K, v);
                    } else {
                        mmm_vect(jj, kk, ll, maxIter, 8, sums, K, v);
                    }
                }
            }
            for (int z = 0; z < 8 * 8; z++) {
                sums[z] = 0;
            }
            int k_start = 8 * (t_gp / 8);


            for (int ll = 0; ll + 7 < k_start; ll += 8) {
                mmm_vect(jj, k_start, ll, maxIter, t_gp - k_start, sums, K, v);
            }

            solve_triangle_vect(X_grid, X, mu, sigma, alpha, i, jj, k_start, k_start, n, maxIter, t_gp - k_start, sums,
                                K, v);

//            for (int k = k_start; k < t_gp; k++) {
//                for (int l = 8 * (k / 8); l < k; ++l) {
//                    for (int j = jj; j < jj + 8; j++) {
//                        sums[(k % 8) * 8 + j % 8] += K[k * maxIter + l] * v[l * 8 + (j % 8)];
//                    }
//                }
//                int x_, y_;
//                float arg1x, arg1y;
//                x_ = X[2 * k];
//                y_ = X[2 * k + 1];
//                arg1x = X_grid[x_ * 2 * n + 2 * y_];
//                arg1y = X_grid[x_ * 2 * n + 2 * y_ + 1];
//                for (int j = jj; j < jj + 8; j++) {
//                    float x_star = X_grid[2 * n * i + 2 * j];
//                    float y_star = X_grid[2 * n * i + 2 * j + 1];
//                    float k_star = expf(
//                            -((arg1x - x_star) * (arg1x - x_star) + (arg1y - y_star) * (arg1y - y_star)) / 2.f);
//                    v[k * 8 + (j % 8)] = (k_star - sums[(k % 8) * 8 + j % 8]) / K[k * maxIter + k];
//                    mu[i * n + j] += k_star * alpha[k];
//                    sigma[i * n + j] -= v[k * 8 + (j % 8)] * v[k * 8 + (j % 8)];
//                }
//            }
            for (int j = jj; j < jj + 8; j++) {
                if (sigma[i * n + j] < 0) {
                    sigma[i * n + j] = 0.0;
                }
                float currentValue = mu[i * n + j] + sqrtf(beta) * sigma[i * n + j];
                if (!sampled[i * n + j] && currentValue > maxValue) {
                    maxValue = currentValue;
                    maxI = i;
                    maxJ = j;
                }
            }
        }
    }
    maxIJ[0] = maxI;
    maxIJ[1] = maxJ;
    free(sums);
    free(x);
    free(alpha);
    free(v);
}