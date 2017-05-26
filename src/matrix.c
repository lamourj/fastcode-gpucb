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
    mmm(0, 0, 0, 0, 8, T, A, B);
}


void run() {
    mmm_vect(0, 0, 0, 0, 8, C, A, B);
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
    printf("Clean done.\n");
}


float frand() {
    return (float) rand() / (float) RAND_MAX;
}

void mmm(int jj, int kk, int ll, int maxIter, int k_max, float *sums, float *K, float *v) {
    for (int j = jj; j < jj + 8; j++) {
        for (int k = kk; k < kk + 8; k++) {
            float tmp_sum = 0;
            for (int l = ll; l < ll + 8; ++l) {
                tmp_sum += K[k * maxIter + l] * v[l * 8 + (j % 8)];
            }
            sums[(k % 8) * 8 + j % 8] += tmp_sum;
        }
    }
}


void mmm_vect(int jj, int kk, int ll, int maxIter, int k_max, float *sums, float *K, float *v) {

    for (int k = kk; k < kk + k_max; k++) {
        const int k_maxIter_ll = k * maxIter + ll;

        __m256 sum = _mm256_loadu_ps(sums + (k % 8) * 8);

        const __m256 k_element_0 = _mm256_set1_ps(K[k_maxIter_ll]);
        const __m256 v_row_0 = _mm256_loadu_ps(v + ll * 8);
        const __m256 v_row_1 = _mm256_loadu_ps(v + (ll + 1) * 8);
        const __m256 k_element_1 = _mm256_set1_ps(K[k_maxIter_ll + 1]);
        const __m256 k_element_2 = _mm256_set1_ps(K[k_maxIter_ll + 2]);
        const __m256 v_row_2 = _mm256_loadu_ps(v + (ll + 2) * 8);

        // l = ll
        sum = _mm256_fmadd_ps(k_element_0, v_row_0, sum);
        const __m256 k_element_3 = _mm256_set1_ps(K[k_maxIter_ll + 3]);
        const __m256 v_row_3 = _mm256_loadu_ps(v + (ll + 3) * 8);

        // l = ll + 1
        sum = _mm256_fmadd_ps(k_element_1, v_row_1, sum);
        const __m256 k_element_4 = _mm256_set1_ps(K[k_maxIter_ll + 4]);
        const __m256 v_row_4 = _mm256_loadu_ps(v + (ll + 4) * 8);

        // l = ll + 2
        sum = _mm256_fmadd_ps(k_element_2, v_row_2, sum);
        const __m256 k_element_5 = _mm256_set1_ps(K[k_maxIter_ll + 5]);
        const __m256 v_row_5 = _mm256_loadu_ps(v + (ll + 5) * 8);

        // l = ll + 3
        sum = _mm256_fmadd_ps(k_element_3, v_row_3, sum);
        const __m256 k_element_6 = _mm256_set1_ps(K[k_maxIter_ll + 6]);
        const __m256 v_row_6 = _mm256_loadu_ps(v + (ll + 6) * 8);

        // l = ll + 4
        sum = _mm256_fmadd_ps(k_element_4, v_row_4, sum);
        const __m256 k_element_7 = _mm256_set1_ps(K[k_maxIter_ll + 7]);
        const __m256 v_row_7 = _mm256_loadu_ps(v + (ll + 7) * 8);

        // l = ll + 5
        sum = _mm256_fmadd_ps(k_element_5, v_row_5, sum);

        // l = ll + 6
        sum = _mm256_fmadd_ps(k_element_6, v_row_6, sum);

        // l = ll + 7
        sum = _mm256_fmadd_ps(k_element_7, v_row_7, sum);

        _mm256_storeu_ps(sums + (k % 8) * 8, sum);
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


int main() {
    initialize(0, 0);
    run();
    clean();
}