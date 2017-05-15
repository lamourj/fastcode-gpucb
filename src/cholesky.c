#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>

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
void incremental_cholesky(float *A, int n1, int n2, int size) {
    for (int i = n1; i < n2; ++i) {
        // Update the off diagonal entries.
        for (int j = 0; j < i; ++j) {
            for (int k = 0; k < j; ++k) {
                A[size * i + j] -= A[size * i + k] * A[size * j + k];
            }
            A[size * i + j] /= A[size * j + j];
        }
        // Update the diagonal entry.
        for (int k = 0; k < i; ++k) {
            A[size * i + i] -= A[size * i + k] * A[size * i + k];
        }
        A[size * i + i] = sqrtf(A[size * i + i]);
    }

}

float frand() {
    return (float) rand() / (float) RAND_MAX * 100;
}


int main() {

    int n = 10;
    float A[n * n];
    float PSD[n * n];
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

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", PSD[n * i + j]);
        }
        printf("\n");
    }

    //cholesky(PSD, n-2, n);

    incremental_cholesky(PSD, 0, n, n);

    gsl_linalg_cholesky_decomp1(L);

    printf("The library decomposition:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", L->data[n * i + j]);
        }
        printf("\n");
    }

    printf("The handwritten decompostion:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", PSD[n * i + j]);
        }
        printf("\n");
    }


    return 0;
}