//
// Created by Daan Nilis on 27/05/2017.
//

#include "triangle.h"
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "immintrin.h"
#include "avx_mathfun.h"

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

void
solve_triangle_vect(float *X_grid, int *X, float *mu, float *sigma, float *alpha, int i, int jj, int kk, int ll, int n,
                    int maxIter, float *sums, float *K, float *v) {
    __m256 mu_vector = _mm256_setzero_ps();
    __m256 sigma_vector = _mm256_setzero_ps();
//    float a_vector[8] = {1, 2, 3, 4, 5, 6, 7};
//    float b_vector[8] = {9, 10, 11, 12, 13, 14, 15};
    const __m256 xy_star0 = _mm256_loadu_ps(X_grid + 2 * n * i + 2 *
                                                                 jj);//_mm256_loadu_ps( X_grid+2 * n * i + 2 * jj); // x_star0,y_star0,x_star1,y_star1,etc._mm256_setr_ps(1,2,3,4,5,6,7, 8)
    const __m256 xy_star1 = _mm256_loadu_ps(X_grid + 2 * n * i + 2 * (jj +
                                                                      4)); // x_star4,y_star4,x_star5,y_star5,etc. // _mm256_setr_ps(9, 10, 11, 12, 13, 14, 15, 16);
    __m256 xstar_h1 = _mm256_permute2f128_ps(xy_star0, xy_star1, 32);
    __m256 xstar_h2 = _mm256_permute2f128_ps(xy_star0, xy_star1, 49);
    const __m256 xstar_vector = _mm256_shuffle_ps(xstar_h1, xstar_h2, 136);
    const __m256 ystar_vector = _mm256_shuffle_ps(xstar_h1, xstar_h2, 221);

    /*printf("x_star0: %lf, x_star1: %lf, x_star2: %lf, x_star3: %lf, x_star4: %lf, x_star5: %lf,x_star6: %lf, x_star7: %lf \n",x_star0,
    x_star1, x_star2, x_star3, x_star4, x_star5, x_star6, x_star7);
    printf("in vector!!!!!!!\n");
    printf("x_star0: %lf, x_star1: %lf, x_star2: %lf, x_star3: %lf, x_star4: %lf, x_star5: %lf,x_star6: %lf, x_star7: %lf\n",xstar_vector[0], xstar_vector[1],
           xstar_vector[2], xstar_vector[3], xstar_vector[4], xstar_vector[5], xstar_vector[6], xstar_vector[7]);
    printf("y_star0: %lf, y_star1: %lf, y_star2: %lf, y_star3: %lf, y_star4: %lf, y_star5: %lf,y_star6: %lf, y_star7: %lf\n",ystar_vector[0], ystar_vector[1],
           ystar_vector[2], ystar_vector[3], ystar_vector[4], ystar_vector[5], ystar_vector[6], ystar_vector[7]);*/

    //const __m256 y_star = _mm256_loadu_ps(X_grid+2 * n * i + 2 * (jj+0) + 1);
    float x_star0 = X_grid[2 * n * i + 2 * (jj + 0) + 0];
    float x_star1 = X_grid[2 * n * i + 2 * (jj + 1) + 0];
    float x_star2 = X_grid[2 * n * i + 2 * (jj + 2) + 0];
    float x_star3 = X_grid[2 * n * i + 2 * (jj + 3) + 0];
    float x_star4 = X_grid[2 * n * i + 2 * (jj + 4) + 0];
    float x_star5 = X_grid[2 * n * i + 2 * (jj + 5) + 0];
    float x_star6 = X_grid[2 * n * i + 2 * (jj + 6) + 0];
    float x_star7 = X_grid[2 * n * i + 2 * (jj + 7) + 0];

    float y_star0 = X_grid[2 * n * i + 2 * (jj + 0) + 1];
    float y_star1 = X_grid[2 * n * i + 2 * (jj + 1) + 1];
    float y_star2 = X_grid[2 * n * i + 2 * (jj + 2) + 1];
    float y_star3 = X_grid[2 * n * i + 2 * (jj + 3) + 1];
    float y_star4 = X_grid[2 * n * i + 2 * (jj + 4) + 1];
    float y_star5 = X_grid[2 * n * i + 2 * (jj + 5) + 1];
    float y_star6 = X_grid[2 * n * i + 2 * (jj + 6) + 1];
    float y_star7 = X_grid[2 * n * i + 2 * (jj + 7) + 1];

    int x_, y_;
    float arg1x, arg1y;

    float mu0 = 0, mu1 = 0, mu2 = 0, mu3 = 0, mu4 = 0, mu5 = 0, mu6 = 0, mu7 = 0;
    float sigma0 = 0, sigma1 = 0, sigma2 = 0, sigma3 = 0, sigma4 = 0, sigma5 = 0, sigma6 = 0, sigma7 = 0;

    for (int k = kk; k < kk + 8; k++) {
        __m256 sum_vector = _mm256_loadu_ps(sums + 8 * k);
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
        //__m256 kstar_vector = _mm256_setr_ps(expf(norm[0]), expf(norm[1]),expf(norm[2]),expf(norm[3]),expf(norm[4]),expf(norm[5]),expf(norm[6]),expf(norm[7]));//_mm256_getexp_ps(norm);
        __m256 kstar_vector = exp256_ps(exponent);
//        printf("kstar vector\n");
//        for (int z = 0; z < 8; z++) {
//            printf("kstar %d: %lf   ", z, kstar_vector[z]);
//        }

        float kstar0 = expf(
                -((arg1x - x_star0) * (arg1x - x_star0) + (arg1y - y_star0) * (arg1y - y_star0)) / 2.f);
        float kstar1 = expf(
                -((arg1x - x_star1) * (arg1x - x_star1) + (arg1y - y_star1) * (arg1y - y_star1)) / 2.f);
        float kstar2 = expf(
                -((arg1x - x_star2) * (arg1x - x_star2) + (arg1y - y_star2) * (arg1y - y_star2)) / 2.f);
        float kstar3 = expf(
                -((arg1x - x_star3) * (arg1x - x_star3) + (arg1y - y_star3) * (arg1y - y_star3)) / 2.f);
        float kstar4 = expf(
                -((arg1x - x_star4) * (arg1x - x_star4) + (arg1y - y_star4) * (arg1y - y_star4)) / 2.f);
        float kstar5 = expf(
                -((arg1x - x_star5) * (arg1x - x_star5) + (arg1y - y_star5) * (arg1y - y_star5)) / 2.f);
        float kstar6 = expf(
                -((arg1x - x_star6) * (arg1x - x_star6) + (arg1y - y_star6) * (arg1y - y_star6)) / 2.f);
        float kstar7 = expf(
                -((arg1x - x_star7) * (arg1x - x_star7) + (arg1y - y_star7) * (arg1y - y_star7)) / 2.f);

//        printf("\n");
//        printf("kstar 0: %lf    ", kstar0);
//        printf("kstar 1: %lf    ", kstar1);
//        printf("kstar 2: %lf    ", kstar2);
//        printf("kstar 3: %lf    ", kstar3);
//        printf("kstar 4: %lf    ", kstar4);
//        printf("kstar 5: %lf    ", kstar5);
//        printf("kstar 6: %lf    ", kstar6);
//        printf("kstar 7: %lf    ", kstar7);
//

        __m256 K_vector;
        __m256 v_vector;
        for (int l = ll; l < k; ++l) {
            K_vector = _mm256_set1_ps(K[k * maxIter + l]);
            v_vector = _mm256_loadu_ps(v + l * 8);
            sums[(k % 8) * 8 + (jj + 0) % 8] += K[k * maxIter + l] * v[l * 8 + ((jj + 0) % 8)];
            sums[(k % 8) * 8 + (jj + 1) % 8] += K[k * maxIter + l] * v[l * 8 + ((jj + 1) % 8)];
            sums[(k % 8) * 8 + (jj + 2) % 8] += K[k * maxIter + l] * v[l * 8 + ((jj + 2) % 8)];
            sums[(k % 8) * 8 + (jj + 3) % 8] += K[k * maxIter + l] * v[l * 8 + ((jj + 3) % 8)];
            sums[(k % 8) * 8 + (jj + 4) % 8] += K[k * maxIter + l] * v[l * 8 + ((jj + 4) % 8)];
            sums[(k % 8) * 8 + (jj + 5) % 8] += K[k * maxIter + l] * v[l * 8 + ((jj + 5) % 8)];
            sums[(k % 8) * 8 + (jj + 6) % 8] += K[k * maxIter + l] * v[l * 8 + ((jj + 6) % 8)];
            sums[(k % 8) * 8 + (jj + 7) % 8] += K[k * maxIter + l] * v[l * 8 + ((jj + 7) % 8)];
            sum_vector = _mm256_fmadd_ps(K_vector, v_vector, sum_vector);
        }

        __m256 K_diag = _mm256_set1_ps(K[k * maxIter + k]);
        v_vector = _mm256_sub_ps(kstar_vector, sum_vector);
        v_vector = _mm256_div_ps(v_vector, K_diag);
        v[k * 8 + ((jj + 0) % 8)] = (kstar0 - sums[(k % 8) * 8 + (jj + 0) % 8]) / K[k * maxIter + k];
        v[k * 8 + ((jj + 1) % 8)] = (kstar1 - sums[(k % 8) * 8 + (jj + 1) % 8]) / K[k * maxIter + k];
        v[k * 8 + ((jj + 2) % 8)] = (kstar2 - sums[(k % 8) * 8 + (jj + 2) % 8]) / K[k * maxIter + k];
        v[k * 8 + ((jj + 3) % 8)] = (kstar3 - sums[(k % 8) * 8 + (jj + 3) % 8]) / K[k * maxIter + k];
        v[k * 8 + ((jj + 4) % 8)] = (kstar4 - sums[(k % 8) * 8 + (jj + 4) % 8]) / K[k * maxIter + k];
        v[k * 8 + ((jj + 5) % 8)] = (kstar5 - sums[(k % 8) * 8 + (jj + 5) % 8]) / K[k * maxIter + k];
        v[k * 8 + ((jj + 6) % 8)] = (kstar6 - sums[(k % 8) * 8 + (jj + 6) % 8]) / K[k * maxIter + k];
        v[k * 8 + ((jj + 7) % 8)] = (kstar7 - sums[(k % 8) * 8 + (jj + 7) % 8]) / K[k * maxIter + k];

        mu_vector = _mm256_fmadd_ps(kstar_vector, _mm256_set1_ps(alpha[k]), mu_vector);
        mu0 += kstar0 * alpha[k];
        mu1 += kstar1 * alpha[k];
        mu2 += kstar2 * alpha[k];
        mu3 += kstar3 * alpha[k];
        mu4 += kstar4 * alpha[k];
        mu5 += kstar5 * alpha[k];
        mu6 += kstar6 * alpha[k];
        mu7 += kstar7 * alpha[k];

        sigma_vector = _mm256_fnmadd_ps(v_vector, v_vector, sigma_vector);
        sigma0 -= v[k * 8 + ((jj + 0) % 8)] * v[k * 8 + ((jj + 0) % 8)];
        sigma1 -= v[k * 8 + ((jj + 1) % 8)] * v[k * 8 + ((jj + 1) % 8)];
        sigma2 -= v[k * 8 + ((jj + 2) % 8)] * v[k * 8 + ((jj + 2) % 8)];
        sigma3 -= v[k * 8 + ((jj + 3) % 8)] * v[k * 8 + ((jj + 3) % 8)];
        sigma4 -= v[k * 8 + ((jj + 4) % 8)] * v[k * 8 + ((jj + 4) % 8)];
        sigma5 -= v[k * 8 + ((jj + 5) % 8)] * v[k * 8 + ((jj + 5) % 8)];
        sigma6 -= v[k * 8 + ((jj + 6) % 8)] * v[k * 8 + ((jj + 6) % 8)];
        sigma7 -= v[k * 8 + ((jj + 7) % 8)] * v[k * 8 + ((jj + 7) % 8)];


    }

    _mm256_storeu_ps(mu + i * n + jj, mu_vector);
    mu[i * n + (jj + 0)] = mu0;
    mu[i * n + (jj + 1)] = mu1;
    mu[i * n + (jj + 2)] = mu2;
    mu[i * n + (jj + 3)] = mu3;
    mu[i * n + (jj + 4)] = mu4;
    mu[i * n + (jj + 5)] = mu5;
    mu[i * n + (jj + 6)] = mu6;
    mu[i * n + (jj + 7)] = mu7;

    _mm256_storeu_ps(sigma + i * n + jj, sigma_vector);
    sigma[i * n + (jj + 0)] = sigma0;
    sigma[i * n + (jj + 1)] = sigma1;
    sigma[i * n + (jj + 2)] = sigma2;
    sigma[i * n + (jj + 3)] = sigma3;
    sigma[i * n + (jj + 4)] = sigma4;
    sigma[i * n + (jj + 5)] = sigma5;
    sigma[i * n + (jj + 6)] = sigma6;
    sigma[i * n + (jj + 7)] = sigma7;
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
}