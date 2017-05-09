#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>

//Straightforward implementation of inplace Cholesky decomposition of matrix A.
void cholesky(float *A, int n) {
    for (int i = 0; i < n; ++i) {

        //Update the off diagonal entries first.
        for (int j = 0; j < i; ++j) {
            for (int k = 0; k < j; ++k) {
                A[n * i + j] -= A[n * i + k] * A[n * j + k];
            }
            A[n * i + j] /= A[n * j + j];
        }

        //Update the diagonal entries.
        for (int k = 0; k < i; ++k) {
            A[n * i + i] -= A[n * i + k] * A[n * i + k];
        }
        A[n * i + i] = sqrtf(A[n * i + i]);
    }
}

float frand() {
    return (float) rand() / (float) RAND_MAX * 100;
}



int main () {

    int n = 10;
    float A[n*n];
    float PSD[n*n];
    gsl_matrix * L = gsl_matrix_alloc(n, n);
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            A[n * i + j] = frand();
        }
    }

    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            PSD[i*n+j] = 0;
            for (int k=0; k<n; ++k) {
                PSD[i*n+j] += A[i*n+k]*A[j*n+k];
            }
            gsl_matrix_set(L, i, j, PSD[i*n+j]);
        }
    }

    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            printf("%f ", PSD[n * i + j]);
        }
        printf("\n");
    }

    cholesky(PSD, n);

    gsl_linalg_cholesky_decomp1(L);

    printf("The library decomposition:\n");
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            printf("%f ", L->data[n * i + j]);
        }
        printf("\n");
    }

    printf("The handwritten decompostion:\n");
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            printf("%f ", PSD[n * i + j]);
        }
        printf("\n");
    }






    return 0;
}