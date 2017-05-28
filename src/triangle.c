//
// Created by Daan Nilis on 27/05/2017.
//

#include "triangle.h"
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "immintrin.h"
#include "avx_mathfun.h"
const char *tag[20] = {"triangle_solve"};

void initialize(const int I, const int N) {
    printf("Init triangle1\n");
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
    maxIJ_ = (int *) malloc(2 * sizeof(int));
    maxIJ_[0] = 0;
    maxIJ_[1] = 0;
     initialize_meshgrid(X_grid_, N_, GRID_MIN_, GRID_INC_);
    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        sampled_[i] = false;
        mu_[i] = 0;
        sigma_[i] = 0.5;
    }

    if (T_ == 0 || X_ == 0 || X_grid_ == 0 || sampled_ == 0 || mu_ == 0 || sigma_ == 0 || K_ == 0 || L_ == 0) {
        printf("ERROR: Out of memory\n");
    }

    for(int j=0; j<2*I;j++)
	X_[j] = 0;


    //initialize(I, n);
    l_ = (float *) malloc(64 * 64 * sizeof(float));
    sums_ = (float *) malloc(8 * 8 * sizeof(float));
    alpha_ = (float *) malloc(I * sizeof(float));
    v_ = (float *) malloc(8*8 * sizeof(float));


    float *A = (float *) malloc(I * I * sizeof(float));

    // gsl_matrix *L = gsl_matrix_alloc(n, n);

    // Make a random PSD matrix:
    for (int i = 0; i < I; ++i) {
        alpha_[i] = frand();
        for (int j = 0; j < I; ++j) {
            A[I * i + j] = frand();
        }
    }

    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < I; ++j) {
            L_[i * I + j] = 0;
            for (int k = 0; k < I; ++k) {
                L_[i * I + j] += A[i * I + k] * A[j * I + k];
            }
        }
    }

    incremental_cholesky(L_, A, 0, I_, I_);
    free(A);
   
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


void clean() {
    free(T_);
    free(X_);
    free(X_grid_);
    free(sampled_);
    free(mu_);
    free(sigma_);
    free(K_);
    free(L_);
    free(alpha_);
    free(l_);
    free(sums_);
    free(v_);
}

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

float frand() {
    return (float) rand() / (float) RAND_MAX;
}

void run(){
    //solve_triangle_vect(X_grid_, X_, mu_, sigma_, alpha_, 0, 0, 0, 0, N_, I_, 8, sums_, L_, v_);
    solve_triangle(X_grid_, X_, mu_, sigma_, alpha_, 0, 0, 0, 0, N_, I_, sums_, L_, v_);
}


/*
int main() {

    int n = 200;
    int I = 50;
    initialize(I, n);
    float v[8 * 8];
    float sums[8 * 8];
    float alpha[I];


    float A[I * I];
    float PSD[I * I];
    // gsl_matrix *L = gsl_matrix_alloc(n, n);

    // Make a random PSD matrix:
    for (int i = 0; i < I; ++i) {
        alpha[i] = frand();
        for (int j = 0; j < I; ++j) {
            A[I * i + j] = frand();
        }
    }

    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < I; ++j) {
            PSD[i * I + j] = 0;
            for (int k = 0; k < I; ++k) {
                PSD[i * I + j] += A[i * I + k] * A[j * I + k];
            }
        }
    }

    incremental_cholesky(PSD, A, 0, I, I);
    solve_triangle_vect(X_grid_, X_, mu_, sigma_, alpha, 0, 0, 0, 0, n, I, 8, sums, PSD, v);
    clean();
}*/
