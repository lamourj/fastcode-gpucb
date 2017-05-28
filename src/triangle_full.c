//
// Created by Daan Nilis on 28/05/2017.
//

#include "triangle_full.h"

void initialize(const int I, const int N) {
    printf("Init solve triangle over full grid\n");
    if (N % 8 != 0) {
        printf("n is not divisible by 8 !!! \n");
    }

    GRID_MIN_ = -6;
    GRID_INC_ = 0.025;

    I_ = I;
    N_ = N;
    X_grid_ = (float *) malloc(2 * N * N * sizeof(float));
    X_ = (int *) malloc(2 * I * sizeof(int));
    mu_ = (float *) malloc(N * N * sizeof(float));
    sigma_ = (float *) malloc(N * N * sizeof(float));
    K_ = (float *) malloc(I * I * sizeof(float));
    x = (float *) malloc(I * sizeof(float));
    alpha = (float *) malloc(I * sizeof(float));
    v = (float *) malloc(8 * I * sizeof(float));
    k_star = (float *) malloc(8 * I * sizeof(float));
    sums = (float *) malloc(64 * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        mu_[i] = 0;
        sigma_[i] = 0.5;
    }

    float A[I*I];

    // Make a PSD matrix:
    for (int i = 0; i < I; ++i) {
        X_[2*i] = 0;
        X_[2*i+1] = 0;
        for (int j = 0; j < I; ++j) {
            A[I * i + j] = frand();
        }
    }

    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < I; ++j) {
            K_[i * I + j] = 0;
            for (int k = 0; k < I; ++k) {
                K_[i * I + j] += A[i * I + k] * A[j * I + k];
            }
        }
    }

    // Do the cholesky decomposition to obtain a triangular system:
    incremental_cholesky(K_, A, 0, I, I);

    if (X_ == 0 || X_grid_ == 0 || mu_ == 0 || sigma_ == 0 || K_ == 0) {
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
    const __m256 v_row_3 = _mm256_loadu_ps(v + (ll + 3) * 8);
    const __m256 v_row_4 = _mm256_loadu_ps(v + (ll + 4) * 8);
    const __m256 v_row_5 = _mm256_loadu_ps(v + (ll + 5) * 8);
    const __m256 v_row_6 = _mm256_loadu_ps(v + (ll + 6) * 8);
    const __m256 v_row_7 = _mm256_loadu_ps(v + (ll + 7) * 8);

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
                         int n, int maxIter, int k_max, float *sums, float *K, float *v, float *k_star) {
    const __m256 xy_star0 = _mm256_loadu_ps(X_grid + 2 * n * i + 2 * jj);
    const __m256 xy_star1 = _mm256_loadu_ps(X_grid + 2 * n * i + 2 * (jj + 4));
    const __m256 xstar_h1 = _mm256_permute2f128_ps(xy_star0, xy_star1, 32);
    __m256 mu_vector = _mm256_loadu_ps(mu + i * n + jj);
    const __m256 xstar_h2 = _mm256_permute2f128_ps(xy_star0, xy_star1, 49);
    __m256 sigma_vector = _mm256_loadu_ps(sigma + i * n + jj);
    const __m256 xstar_vector = _mm256_shuffle_ps(xstar_h1, xstar_h2, 136);
    const __m256 minus_two_vector = _mm256_set1_ps(-2.f);
    const __m256 ystar_vector = _mm256_shuffle_ps(xstar_h1, xstar_h2, 221);

    int x_, y_;
    for (int k = kk; k < kk + k_max; k++) {
        __m256 sum_vector = _mm256_loadu_ps(sums + 8 * (k % 8));
        x_ = X[2 * k];
        y_ = X[2 * k + 1];
        const float arg1x = X_grid[x_ * 2 * n + 2 * y_];
        const float arg1y = X_grid[x_ * 2 * n + 2 * y_ + 1];
        const __m256 arg1x_vector = _mm256_set1_ps(arg1x);
        const __m256 arg1y_vector = _mm256_set1_ps(arg1y);
        const __m256 exponentx_0 = _mm256_sub_ps(arg1x_vector, xstar_vector);
        const __m256 exponentx = _mm256_mul_ps(exponentx_0, exponentx_0);
        const __m256 exponenty_0 = _mm256_sub_ps(arg1y_vector, ystar_vector);
        const __m256 exponenty = _mm256_mul_ps(exponenty_0, exponenty_0);
        const __m256 exponentxy = _mm256_add_ps(exponentx, exponenty);
        const __m256 exponent = _mm256_div_ps(exponentxy, minus_two_vector);
        const __m256 kstar_vector_new = exp256_ps(exponent);
        const __m256 old_kstar = _mm256_loadu_ps(k_star + k * 8);
        const __m256 kstar_vector = _mm256_add_ps(old_kstar, kstar_vector_new);

        __m256 v_vector_0;
        for (int l = ll; l < k; ++l) {
            const __m256 K_vector = _mm256_set1_ps(K[k * maxIter + l]);
            v_vector_0 = _mm256_loadu_ps(v + l * 8);
            sum_vector = _mm256_fmadd_ps(K_vector, v_vector_0, sum_vector);
        }

        _mm256_storeu_ps(sums + (k % 8) * 8, sum_vector);

        const __m256 K_diag = _mm256_set1_ps(K[k * maxIter + k]);
        const __m256 v_vector_1 = _mm256_sub_ps(kstar_vector, sum_vector);
        const __m256 v_vector_2 = _mm256_div_ps(v_vector_1, K_diag);
        _mm256_storeu_ps(v + k * 8, v_vector_2);

        mu_vector = _mm256_fmadd_ps(kstar_vector, _mm256_set1_ps(alpha[k]), mu_vector);

        sigma_vector = _mm256_fnmadd_ps(v_vector_2, v_vector_2, sigma_vector);

    }
    _mm256_storeu_ps(mu + i * n + jj, mu_vector);
    _mm256_storeu_ps(sigma + i * n + jj, sigma_vector);
}


void run() {

    int n = N_;

    int t_gp = I_;
    int maxIter = I_;

    const int t_gp_8 = 8 * t_gp;
    const int k_start = 8 * (t_gp / 8);
    const int k_start_minus_7 = k_start - 7;


    for (int i = 0; i < n; i++) { // for all points in X_grid ([i])
        const int in = i * n;
        for (int jj = 0; jj < n; jj += 8) { // for all points in X_grid ([i][j])
            for (int j = jj; j < jj + 8; j++) {
                const int inj = in + j;
                mu[inj] = 0;
                sigma[inj] = 1.0;
            }
            for (int zz = 0; zz < t_gp_8; ++zz) {
                v[zz] = 0;
                k_star[zz] = 0;
            }
            for (int kk = 0; kk + 7 < t_gp; kk += 8) {
                for (int z = 0; z < 8 * 8; z++) {
                    sums[z] = 0;
                }
                for (int ll = 0; ll <= kk; ll += 8) {
                    if (ll == kk) {
//                        solve_triangle_vect(X_grid, X_, mu_, sigma_, alpha, i, jj, kk, ll, n, maxIter, 8, sums, K, v,
//                                            k_star);
                        solve_triangle(X_grid, X_, mu_, sigma_, alpha, i, jj, kk, ll, n, maxIter, sums, K, v);
                    } else {
//                        mmm_vect(jj, kk, ll, maxIter, 8, sums, K, v);
                        mmm(jj, kk, ll, maxIter, 8, sums, K_, v);
                    }
                }
            }
            for (int z = 0; z < 64; z++) {
                sums[z] = 0;
            }

            for (int ll = 0; ll < k_start_minus_7; ll += 8) {
//                mmm_vect(jj, k_start, ll, maxIter, t_gp - k_start, sums, K, v);
                mmm(jj, k_start, ll, maxIter, t_gp-k_start, sums, K, v);
            }
//            solve_triangle_vect(X_grid, X_, mu_, sigma_, alpha, i, jj, k_start, k_start, n, maxIter, t_gp - k_start, sums,
//                                K, v, k_star);
            solve_triangle(X_grid, X_, mu_, sigma_, alpha, i, jj, kk, ll, n, maxIter, sums, K, v);
        }
    }
}