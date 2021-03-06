#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
// #include <gsl/gsl_linalg.h>
// #include <gsl/gsl_matrix.h>
#include <x86intrin.h>
#include "immintrin.h"


float kernel2_baseline(float *x1, float *y1, float *x2, float *y2) {
    // RBF kernel
    float sigma = 1;
    return (float) expf(-((*x1 - *x2) * (*x1 - *x2) + (*y1 - *y2) * (*y1 - *y2)) / (float) (2 * sigma * sigma));
}


void kernel2_baseline_vect(__m256 x1, __m256 y1, __m256 x2, __m256 y2, float *result) {
    // RBF kernel
    __m256 sigma = _mm256_set1_ps(2.0);
    x1 = _mm256_sub_ps(x1, x2);
    y1 = _mm256_sub_ps(y1, y2);
    x1 = _mm256_mul_ps(x1, x1);
    y1 = _mm256_fmadd_ps(y1, y1, x1);
    y1 = _mm256_sub_ps(_mm256_set1_ps(0.0), y1);

    y1 = _mm256_div_ps(y1, sigma);
    for (int i = 0; i < 8; ++i) {
        result[i] = expff(y1[i]);
    }
}

/*
 Straightforward implementation of inplace Cholesky decomposition of matrix A.
 Input arguments:
    A:    The matrix to decompose
    n:    The size of the data in matrix A to decompose
    size: The actual size of the rows
 */
void cholesky(float *A, float *result, int n, int size) {
    for (int i = 0; i < n; ++i) {

        // Update the off diagonal entries first.
        for (int j = 0; j < i; ++j) {
            result[size * i + j] = A[size * i + j];
            for (int k = 0; k < j; ++k) {
                result[size * i + j] -= result[size * i + k] * result[size * j + k];
            }
            result[size * i + j] /= result[size * j + j];
        }

        // Update the diagonal entry of this row.
        result[size * i + i] = A[size * i + i];
        for (int k = 0; k < i; ++k) {
            result[size * i + i] -= result[size * i + k] * result[size * i + k];
        }
        result[size * i + i] = sqrtff(result[size * i + i]);
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
void incremental_cholesky(float *A, float *result, int n1, int n2, int size) {
    for (int i = n1; i < n2; ++i) {
        // Update the off diagonal entries.
        for (int j = 0; j < i; ++j) {
            result[size * i + j] = A[size * i + j];
            for (int k = 0; k < j; ++k) {
                result[size * i + j] -= result[size * i + k] * result[size * j + k];
            }
            result[size * i + j] /= result[size * j + j];
        }
        // Update the diagonal entry.
        result[size * i + i] = A[size * i + i];
        for (int k = 0; k < i; ++k) {
            result[size * i + i] -= result[size * i + k] * result[size * i + k];
        }
        result[size * i + i] = sqrtff(result[size * i + i]);
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
void incremental_cholesky_vect(float *A, float *result, int n1, int n2, int size) {
    __m256 v1, v2, acc;

    float acc1;
    for (int i = n1; i < n2; ++i) {

        // Update the off diagonal entries.

        for (int j = 0; j < i; ++j) {

            int k;
            acc = _mm256_setzero_ps();
            acc1 = 0.0;

            for (k = 0; k + 7 < j; k += 8) {
                v1 = _mm256_loadu_ps(&result[size * j + k]);
                v2 = _mm256_loadu_ps(&result[size * i + k]);
                acc = _mm256_fmadd_ps(v1, v2, acc);
            }
            for (; k < j; ++k) {
                acc1 += result[size * i + k] * result[size * j + k];
            }

            result[size * i + j] =
                    (A[size * i + j] - acc1 - acc[0] - acc[1] - acc[2] - acc[3] - acc[4] - acc[5] - acc[6] - acc[7]) /
                    result[size * j + j];
        }

        // Update the diagonal entry.

        result[size * i + i] = A[size * i + i];
        int k;
        acc = _mm256_setzero_ps();
        acc1 = 0.0;
        for (k = 0; k + 7 < i; k += 8) {
            v1 = _mm256_loadu_ps(&result[size * i + k]);
            acc = _mm256_fmadd_ps(v1, v1, acc);
            //result[size * i + i] -= result[size * i + k] * result[size * i + k];
        }
        while (k < i) {
            acc1 += result[size * i + k] * result[size * i + k];
            k += 1;
        }
        result[size * i + i] = sqrtff(
                A[size * i + i] - acc[0] - acc[1] - acc[2] - acc[3] - acc[4] - acc[5] - acc[6] - acc[7] - acc1);
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

void cholesky_solve2_opt(int d, int size, float *LU, float *b, float *x, int lower) {

    if (lower == 1) {
        float sum0 = 0.;
        for (int i = 0; i < d; ++i) {
            float sum1 = 0.;
            float sum2 = 0.;
            float sum3 = 0.;

            for (int k = 0; k + 3 < i; k += 4) {
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
            float sum0 = 0.;
            float sum1 = 0.;
            float sum2 = 0.;
            float sum3 = 0.;

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


/*
 * Solver for a matrix that is in Cholesky decomposition.
 * Input arguments:
 *      d: dimension of matrix
 *      size: the actual size of the matrix
 *      LU: matrix
 *      b: right hand side
 *      x: matrix (d * 8) to put result in
 *      lower: if one the lower triangle system is solved, else the upper triangle system is solved.
*/
void cholesky_solve2_vect(int d, int size, float *LU, float *b0, float *x, int lower) {
    __m256 vsum, vsum1, vsum2, vsum3, vsum4, v1, v2, v3, v4, v5;
    if (lower == 1) {
        for (int i = 0; i < d; ++i) {
            vsum1 = _mm256_setzero_ps();
            vsum2 = _mm256_setzero_ps();
            vsum3 = _mm256_setzero_ps();
            vsum4 = _mm256_setzero_ps();
            int k;
            for (k = 0; k < i; k += 1) {
                v1 = _mm256_set1_ps(LU[i * size + k]);
//                v2 = _mm256_set1_ps(LU[i * size + k + 1]);
//                v3 = _mm256_set1_ps(LU[i * size + k + 2]);
//                v4 = _mm256_set1_ps(LU[i * size + k + 3]);
                v5 = _mm256_loadu_ps(&x[8 * k]);


                vsum1 = _mm256_fmadd_ps(v1, v5, vsum1);
//                vsum2 = _mm256_fmadd_ps(v2, v5, vsum2);
//                vsum3 = _mm256_fmadd_ps(v3, v5, vsum3);
//                vsum4 = _mm256_fmadd_ps(v4, v5, vsum4);
            }
//            vsum1 = _mm256_add_ps(vsum1, vsum2);
//            vsum3 = _mm256_add_ps(vsum3, vsum4);
//            vsum1 = _mm256_add_ps(vsum1, vsum3);
            while (k < i) {
                v1 = _mm256_set1_ps(LU[i * size + k]);
                v5 = _mm256_loadu_ps(&x[8 * k]);

                vsum1 = _mm256_fmadd_ps(v1, v5, vsum1);
                k += 1;
            }
            __m256 b = _mm256_loadu_ps(&b0[8 * i]);
            vsum1 = _mm256_sub_ps(b, vsum1);
            __m256 divisor = _mm256_set1_ps(LU[i * size + i]);
            _mm256_storeu_ps(&x[8 * i], _mm256_div_ps(vsum1, divisor));
//            x[i * 8    ] = (b0[i] - vsum1[0]) / LU[i * size + i];
//            x[i * 8 + 1] = (b1[i] - vsum1[1]) / LU[i * size + i];
//            x[i * 8 + 2] = (b2[i] - vsum1[2]) / LU[i * size + i];
//            x[i * 8 + 3] = (b3[i] - vsum1[3]) / LU[i * size + i];
//            x[i * 8 + 4] = (b4[i] - vsum1[4]) / LU[i * size + i];
//            x[i * 8 + 5] = (b5[i] - vsum1[5]) / LU[i * size + i];
//            x[i * 8 + 6] = (b6[i] - vsum1[6]) / LU[i * size + i];
//            x[i * 8 + 7] = (b7[i] - vsum1[7]) / LU[i * size + i];
        }
    } else {
        for (int i = d - 1; i >= 0; --i) {
            vsum = _mm256_setzero_ps();
            for (int k = i + 1; k < d; ++k) {
                v1 = _mm256_set1_ps(LU[i * size + k]);
                v2 = _mm256_loadu_ps(&x[4 * k]);
                vsum = _mm256_fmadd_ps(v1, v2, vsum);
            }
//            x[i * 4    ] = (b0[i] - vsum[0]) / LU[i * size + i];
//            x[i * 4 + 1] = (b1[i] - vsum[1]) / LU[i * size + i];
//            x[i * 4 + 2] = (b2[i] - vsum[2]) / LU[i * size + i];
//            x[i * 4 + 3] = (b3[i] - vsum[3]) / LU[i * size + i];
        }
    }
}

float frand() {
    return (float) rand() / (float) RAND_MAX;
}

const char *tag[50] = {"cholesky_solve2_opt"};

int d;
int size;
float *LU;
float *result;
float *b;
float *b_smal;
float *x;
float *x_smal;


void initialize(const int I, const int N) {
    d = I;
    size = I;
    LU = malloc(sizeof(float) * I * I);
    result = malloc(sizeof(float) * I * I);
    b = malloc(sizeof(float) * 8 * I);
    b_smal = malloc(sizeof(float) * I);
    x = malloc(sizeof(float) * 8 * I);
    x_smal = malloc(sizeof(float) * I);

    int n = I;
    float A[I * I];
    // Make a random PSD matrix:
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[n * i + j] = frand();
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            LU[i * n + j] = 0;
            for (int k = 0; k < n; ++k) {
                LU[i * n + j] += A[i * n + k] * A[j * n + k];
            }
        }
    }

    // Make the righthand side
    for (float i = 0; i < n * 8; ++i) {
        int j = (int) i;
        b[j] = sin(i);
    }

    for (float i = 0; i < n; ++i) {
        int j = (int) i;
        b_smal[j] = sin(i);
    }
}


void run() {
    // for (int i=0; i<8; ++i) {
    //     cholesky_solve2(d, size, LU, b_smal, x, 1);
    // }
    cholesky_solve2_opt(d, size, LU, b_smal, x_smal, 1);
}

void clean() {
    free(LU);
    free(result);
    free(b);
    free(b_smal);
    free(x);
    free(x_smal);
}

/*
int main() {

    int n = 10;
    int gridsize = 9;


    float A[n * n];
    float result[n * n];
    float PSD[n * n];
    // gsl_matrix *L = gsl_matrix_alloc(n, n);

    // Make a random PSD matrix:
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[n * i + j] = frand();
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            PSD[i * n + j] = 0;
            for (int k = 0; k < n; ++k) {
                PSD[i * n + j] += A[i * n + k] * A[j * n + k];
            }
        }
    }

    // Initialize the grid:
    float X_grid[gridsize * gridsize * 2];
    float min = -5;
    float inc = 0.1;

    float x = min;
    for (int i = 0; i < gridsize; i++) {
        float y = min;
        for (int j = 0; j < gridsize; j++) {
            X_grid[i * 2 * gridsize + 2 * j] = y;
            X_grid[i * 2 * gridsize + 2 * j + 1] = x;
            y += inc;
        }
        x += inc;
    }


    // Initialize the sampled points:
    float b[n];
    for (float i = 0; i < n; ++i) {
        int j = (int) i;
        b[j] = sin(i);
    }
    float v[n];



    // Validation:


    cholesky_solve2(n, n, PSD, b, v, 1);
    printf("The vectorized version\n");
    for (int i = 0; i < 8; ++i) {
        printf("%f ", v[i]);
    }


    cholesky_solve2_opt(n, n, PSD, b, v, 1);

    printf("The vectorized version\n");
    for (int i = 0; i < 8; ++i) {
        printf("%f ", v[i]);
    }
}
*/
/*

    // Run the timing expferiment for different grid sizes:
    for (int i = 8; i < gridsize; i+=8) {

        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                gsl_matrix_set(L, j, k, PSD[j * n + k]);
            }
        }


        clock_t start, end;
        float CPU_time, CPU_optimized;


        incremental_cholesky(PSD, result, 0, n, n);


        float v[n];

        start = clock();

        // Time cholesky decomposition:
//        for (int t=0; t<10000; t++) {
//            incremental_cholesky(PSD, result, 0, i, n);
//        }


        // Time system solve:
        for (int t=0; t<1; ++t) {
            for (int l = 0; l < i; l++) // for all points in X_grid
            {
                for (int j = 0; j < i; j++) // for all points in X_grid
                {
                    float x_star = X_grid[2 * gridsize * l + 2 * j]; // Current grid point that we are looking at
                    float y_star = X_grid[2 * gridsize * l + 2 * j + 1];
                    float k_star[n];

                    for (int k = 0; k < n; k++) {
                        int x = X[2 * k];
                        int y = X[2 * k + 1];
                        float arg1x = X_grid[x * 2 * gridsize + 2 * y];
                        float arg1y = X_grid[x * 2 * gridsize + 2 * y + 1];
                        k_star[k] = kernel2_baseline(&arg1x, &arg1y, &x_star, &y_star);
                    }

                    cholesky_solve2(n, n, result, k_star, v, 1);
                    float sum = 0;
                    for (int k=0; k<n; ++k) {
                        sum += v[k];
                    }
                    printf("%f ", sum);
                }
                printf("\n");
            }
        }

        end = clock();
        CPU_time = (end - start);

        start = clock();
        // Time cholesky decomposition:
//        for(int t=0; t<10000; t++) {
//            cholesky(PSD, result, i, n);
//        }


        printf("The vectorized version:\n");
        // Time system solve
        for (int t=0; t<1; ++t) {
            for (int l = 0; l < i; l++) // for all points in X_grid
            {
                for (int j = 0; j < i; j += 8) // for all points in X_grid
                {
                    // float x_star0, x_star1, x_star2, x_star3, x_star4, x_star5, x_star6, x_star7;
                    // float y_star0, y_star1, y_star2, y_star3, y_star4, y_star5, y_star6, y_star7;
                    // float k_star0[n], k_star1[n], k_star2[n], k_star3[n], k_star4[n], k_star5[n], k_star6[n], k_star7[n];
                    float k_star[n * 8];
                    // x_star0 = X_grid[2 * gridsize * l + 2 * (j)];
                    // y_star0 = X_grid[2 * gridsize * l + 2 * (j) + 1];
                    // x_star1 = X_grid[2 * gridsize * l + 2 * (j + 1)];
                    // y_star1 = X_grid[2 * gridsize * l + 2 * (j + 1) + 1];
                    // x_star2 = X_grid[2 * gridsize * l + 2 * (j + 2)];
                    // y_star2 = X_grid[2 * gridsize * l + 2 * (j + 2) + 1];
                    // x_star3 = X_grid[2 * gridsize * l + 2 * (j + 3)];
                    // y_star3 = X_grid[2 * gridsize * l + 2 * (j + 3) + 1];
                    // x_star4 = X_grid[2 * gridsize * l + 2 * (j + 4)];
                    // y_star4 = X_grid[2 * gridsize * l + 2 * (j + 4) + 1];
                    // x_star5 = X_grid[2 * gridsize * l + 2 * (j + 5)];
                    // y_star5 = X_grid[2 * gridsize * l + 2 * (j + 5) + 1];
                    // x_star6 = X_grid[2 * gridsize * l + 2 * (j + 6)];
                    // y_star6 = X_grid[2 * gridsize * l + 2 * (j + 6) + 1];
                    // x_star7 = X_grid[2 * gridsize * l + 2 * (j + 7)];
                    // y_star7 = X_grid[2 * gridsize * l + 2 * (j + 7) + 1];

                    __m256 v1, v2, x_star, y_star;

                    v1 = _mm256_loadu_ps(&X_grid[2 * gridsize * l + 2 * j]);
                    v2 = _mm256_loadu_ps(&X_grid[2 * gridsize * l + 2 * (j + 4)]);

                    // Build the mask for the blend function

                    x_star = _mm256_blend_ps(v1, v2, 170);
                    y_star = _mm256_blend_ps(v1, v2, 85);

                    __m256 arg1x, arg1y;
                    for (int kk = 0; kk < n; kk++) {
                        int x = X[2 * kk];
                        int y = X[2 * kk + 1];
                        arg1x = _mm256_set1_ps(X_grid[x * 2 * gridsize + 2 * y]);
                        arg1y = _mm256_set1_ps(X_grid[x * 2 * gridsize + 2 * y + 1]);
                        // k_star0[kk] = kernel2_baseline(&arg1x, &arg1y, &x_star0, &y_star0);
                        // k_star1[kk] = kernel2_baseline(&arg1x, &arg1y, &x_star1, &y_star1);
                        // k_star2[kk] = kernel2_baseline(&arg1x, &arg1y, &x_star2, &y_star2);
                        // k_star3[kk] = kernel2_baseline(&arg1x, &arg1y, &x_star3, &y_star3);
                        // k_star4[kk] = kernel2_baseline(&arg1x, &arg1y, &x_star4, &y_star4);
                        // k_star5[kk] = kernel2_baseline(&arg1x, &arg1y, &x_star5, &y_star5);
                        // k_star6[kk] = kernel2_baseline(&arg1x, &arg1y, &x_star6, &y_star6);
                        // k_star7[kk] = kernel2_baseline(&arg1x, &arg1y, &x_star7, &y_star7);
                        kernel2_baseline_vect(arg1x, arg1y, x_star, y_star, &k_star[kk * 8]);
                    }
                    float v[n*8];
                    cholesky_solve2_vect(n, n, result, k_star, v, 1);
                    for (int k=0; k<8; ++k) {
                        float sum = 0;
                        for (int m=0; m<n; ++m) {
                            sum += v[m * 8 + k];
                        }
                        printf("%f ", sum);
                    }
                    printf("\n");
                }
            }
        }

        end = clock();
        CPU_optimized = (end - start);
        printf("Execution time for %d grid points: %lf\n",i*i, CPU_time/CPU_optimized);

        gsl_linalg_cholesky_decomp1(L);

//        printf("The library decomposition:\n");
//        for (int ii= 0; ii< n; ++ii) {
//            for (int j = 0; j < n; ++j) {
//                printf("%f ", L->data[n * ii+ j]);
//            }
//            printf("\n");
//        }
//
//        printf("The handwritten decompostion:\n");
//        for (int ii= 0; ii< n; ++ii) {
//            for (int j = 0; j < n; ++j) {
//                printf("%f ", result[n * ii+ j]);
//            }
//            printf("\n");
//        }
    }


//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            printf("%f ", PSD[n * i + j]);
//        }
//        printf("\n");
//    }

    //cholesky(PSD, n-2, n);



    return 0;
}

*/