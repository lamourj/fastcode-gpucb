#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <x86intrin.h>


/*
 Straightforward implementation of inplace Cholesky decomposition of matrix A.
 Input arguments:
    A:    The matrix to decompose
    n:    The size of the data in matrix A to decompose
    size: The actual size of the rows
 */
void cholesky(double *A, int n, int size) {
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
void incremental_cholesky(double *A, double *result, int n1, int n2, int size) {
    __m256d v1, v2, acc;

    double acc1;
    for (int i = n1; i < n2; ++i) {

        // Update the off diagonal entries.

        for (int j = 0; j < i; ++j) {

            int k;
            acc = _mm256_setzero_pd();
            acc1 = 0.0;

            for (k = 0; k + 3 < j; k += 4) {
                v1 = _mm256_loadu_pd(&result[size * j + k]);
                v2 = _mm256_loadu_pd(&result[size * i + k]);
                acc = _mm256_fmadd_pd(v1, v2, acc);
            }

            while (k < j) {
                acc1 += result[size * i + k] * result[size * j + k];
                k+=1;
            }

            result[size * i + j] = (A[size * i + j] - acc1 - acc[0] - acc[1] - acc[2] - acc[3])/result[size * j + j];
        }

        // Update the diagonal entry.

        result[size * i + i] = A[size * i + i];
        int k;
        acc = _mm256_setzero_pd();
        acc1 = 0.0;
        for (k = 0; k + 3< i; k+=4) {
            v1 = _mm256_loadu_pd(&result[size * i + k]);
            acc = _mm256_fmadd_pd(v1, v1, acc);
            //result[size * i + i] -= result[size * i + k] * result[size * i + k];
        }
        while (k < i) {
            acc1 += result[size * i + k] * result[size * i + k];
            k+=1;
        }
        result[size * i + i] = sqrtf(A[size * i + i] - acc[0] - acc[1] - acc[2] - acc[3] - acc1);
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
void cholesky_solve2(int d, int size, double *LU, double *b, double *x, int lower) {
    if (lower == 1) {
        for (int i = 0; i < d; ++i) {
            double sum = 0.;
            for (int k = 0; k < i; ++k) {
                sum += LU[i * size + k] * x[k];
            }
            x[i] = (b[i] - sum) / LU[i * size + i];
        }
    } else {
        for (int i = d - 1; i >= 0; --i) {
            double sum = 0.;
            for (int k = i + 1; k < d; ++k) {
                sum += LU[i * size + k] * x[k];
            }
            x[i] = (b[i] - sum) / LU[i * size + i];
        }
    }

}


double frand() {
    return (double) rand() / (double) RAND_MAX * 100;
}


int main() {

    int n = 100;
    double A[n * n];
    double result[n * n];
    double PSD[n * n];
    gsl_matrix *L = gsl_matrix_alloc(n, n);
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
            gsl_matrix_set(L, i, j, PSD[i * n + j]);
        }
    }

//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            printf("%f ", PSD[n * i + j]);
//        }
//        printf("\n");
//    }

    //cholesky(PSD, n-2, n);

    clock_t start, end;
    double CPU_time;

    start = clock();
    for (int t=0; t<10000; t++) {
        incremental_cholesky(PSD, result, 0, n, n);
    }
    end = clock();
    CPU_time = (end - start);
    printf("execution time: %lf\n", CPU_time/10000);

    gsl_linalg_cholesky_decomp1(L);

//    printf("The library decomposition:\n");
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            printf("%f ", L->data[n * i + j]);
//        }
//        printf("\n");
//    }
//
//    printf("The handwritten decompostion:\n");
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            printf("%f ", result[n * i + j]);
//        }
//        printf("\n");
//    }


    return 0;
}