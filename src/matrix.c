#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "matrix.h"
#include <immintrin.h>


const char *tag[10] = {"matrix"};

float *A, *B, *C, *T;

void initialize(const int I, const int N) {
    A = (float *) malloc(64 * sizeof(float));
    B = (float *) malloc(64 * sizeof(float));
    C = (float *) malloc(64 * sizeof(float));
    T = (float *) malloc(64 * sizeof(float));

    for (int i = 0; i < 64; i++) {
        A[i] = frand();
        B[i] = frand();
        C[i] = 0;
        T[i] = 0;
    }

    float sum = 0;
    mmm(0, 0, 0, 0, 0, T, A, B);
}


void run() {
    mmm_vect(0, 0, 0, 0, 0, C, A, B);
}

void clean() {
    for (int i = 0; i < 64; i++) {
        if (fabsf(C[i] - T[i]) > 0.0001) {
            printf("Different result in C and T: %f %f\n", C[i], T[i]);
        }
    }

    free(A);
    free(B);
    free(C);
    free(T);
}


float frand() {
    return (float) rand() / (float) RAND_MAX;
}

void mmm_vect(int jj, int kk, int ll, int maxIter, int t_gp, float *sums, float *K, float *v) {
    const int ll8 = ll * 8;
    const int kk80_mod_8 = (kk % 8) * 8;
    const int kk81_mod_8 = ((kk + 1) % 8) * 8;
    const int kk82_mod_8 = ((kk + 2) % 8) * 8;
    const int kk83_mod_8 = ((kk + 3) % 8) * 8;
    const int kk84_mod_8 = ((kk + 4) % 8) * 8;
    const int kk85_mod_8 = ((kk + 5) % 8) * 8;
    const int kk86_mod_8 = ((kk + 6) % 8) * 8;
    const int kk87_mod_8 = ((kk + 7) % 8) * 8;

    const __m256 k_vector0 = _mm256_loadu_ps(K + kk * maxIter + ll);
    const __m256 k_vector1 = _mm256_loadu_ps(K + (kk + 1) * maxIter + ll);
    const __m256 k_vector2 = _mm256_loadu_ps(K + (kk + 2) * maxIter + ll);
    const __m256 k_vector3 = _mm256_loadu_ps(K + (kk + 3) * maxIter + ll);
    const __m256 k_vector4 = _mm256_loadu_ps(K + (kk + 4) * maxIter + ll);
    const __m256 k_vector5 = _mm256_loadu_ps(K + (kk + 5) * maxIter + ll);
    const __m256 k_vector6 = _mm256_loadu_ps(K + (kk + 6) * maxIter + ll);
    const __m256 k_vector7 = _mm256_loadu_ps(K + (kk + 7) * maxIter + ll);

    for (int j = jj; j < jj + 8; j++) {
        const int j_mod_8 = j % 8;
        const float sum_0 = sums[kk80_mod_8 + j_mod_8];

        const __m256 v_vector = _mm256_loadu_ps(v + j_mod_8 * t_gp + ll);

        // k = kk
        const __m256 k_v0 = _mm256_mul_ps(k_vector0, v_vector);
        const float sum_1 = sums[kk81_mod_8 + j_mod_8]; // load for next it
        const float sum_tmp_0 = hsum_mm256(k_v0);
        const float new_sum_0 = sum_0 + sum_tmp_0;
        sums[kk80_mod_8 + j_mod_8] = new_sum_0;

        // k = kk + 1
        const __m256 k_v1 = _mm256_mul_ps(k_vector1, v_vector);
        const float sum_2 = sums[kk82_mod_8 + j_mod_8]; // load for next it
        const float tmp_sum_1 = hsum_mm256(k_v1);
        const float new_sum_1 = sum_1 + tmp_sum_1;
        sums[kk81_mod_8 + j_mod_8] = new_sum_1;


        // k = kk + 2
        const __m256 k_v2 = _mm256_mul_ps(k_vector2, v_vector);
        const float sum_3 = sums[kk83_mod_8 + j_mod_8]; // load for next it
        const float tmp_sum_2 = hsum_mm256(k_v2);
        const float new_sum_2 = sum_2 + tmp_sum_2;
        sums[kk82_mod_8 + j_mod_8] = new_sum_2;

        // k = kk + 3
        const __m256 k_v3 = _mm256_mul_ps(k_vector3, v_vector);
        const float sum_4 = sums[kk84_mod_8 + j_mod_8];// load for next it
        const float tmp_sum_3 = hsum_mm256(k_v3);
        const float new_sum_3 = sum_3 + tmp_sum_3;
        sums[kk83_mod_8 + j_mod_8] = new_sum_3;

        // k = kk + 4
        const __m256 k_v4 = _mm256_mul_ps(k_vector4, v_vector);
        const float sum_5 = sums[kk85_mod_8 + j_mod_8];// load for next it
        const float tmp_sum_4 = hsum_mm256(k_v4);
        const float new_sum_4 = sum_4 + tmp_sum_4;
        sums[kk84_mod_8 + j_mod_8] = new_sum_4;

        // k = kk + 5
        const __m256 k_v5 = _mm256_mul_ps(k_vector5, v_vector);
        const float sum_6 = sums[kk86_mod_8 + j_mod_8]; // load for next it
        const float tmp_sum_5 = hsum_mm256(k_v5);
        const float new_sum_5 = sum_5 + tmp_sum_5;
        sums[kk85_mod_8 + j_mod_8] = new_sum_5;

        // k = kk + 6
        const __m256 k_v6 = _mm256_mul_ps(k_vector6, v_vector);
        const float sum_7 = sums[kk87_mod_8 + j_mod_8]; // load for next it
        const float tmp_sum_6 = hsum_mm256(k_v6);
        const float new_sum_6 = sum_6 + tmp_sum_6;
        sums[kk86_mod_8 + j_mod_8] = new_sum_6;

        // k = kk + 7
        const __m256 k_v7 = _mm256_mul_ps(k_vector7, v_vector);
        const float tmp_sum_7 = hsum_mm256(k_v7);
        const float new_sum_7 = sum_7 + tmp_sum_7;
        sums[kk87_mod_8 + j_mod_8] = new_sum_7;
    }
}

float hsum_mm256(__m256 x) {
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 lo = _mm256_extractf128_ps(x, 0);
    lo = _mm_add_ps(hi, lo);
    hi = _mm_movehl_ps(hi, lo);
    lo = _mm_add_ps(hi, lo);
    hi = _mm_shuffle_ps(lo, lo, 1);
    lo = _mm_add_ss(hi, lo);
    return _mm_cvtss_f32(lo);
}

void mmm(int jj, int kk, int ll, int maxIter, int t_gp, float *sums, float *K, float *v) {
    for (int j = jj; j < jj + 8; j++) {

        for (int k = kk; k < kk + 8; k++) {
            float tmp_sum = 0;
            for (int l = ll; l < ll + 8; ++l) {
                tmp_sum += K[k * maxIter + l] * v[(j % 8) * t_gp + l];
            }
            sums[(k % 8) * 8 + j % 8] += tmp_sum;
        }

    }
}

/*int main() {
    initialize(0, 0);
    run();
    clean();
}*/