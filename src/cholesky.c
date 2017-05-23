#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <x86intrin.h>


double kernel2_baseline(double *x1, double *y1, double *x2, double *y2) {
    // RBF kernel
    double sigma = 1;
    return exp(-((*x1 - *x2) * (*x1 - *x2) + (*y1 - *y2) * (*y1 - *y2)) / (2 * sigma * sigma));
}

/*
 Straightforward implementation of inplace Cholesky decomposition of matrix A.
 Input arguments:
    A:    The matrix to decompose
    n:    The size of the data in matrix A to decompose
    size: The actual size of the rows
 */
void cholesky(double *A, double  *result, int n, int size) {
    for (int i = 0; i < n; ++i) {

        // Update the off diagonal entries first.
        for (int j = 0; j < i; ++j) {
            result[size*i+j] = A[size*i+j];
            for (int k = 0; k < j; ++k) {
                result[size * i + j] -= result[size * i + k] * result[size * j + k];
            }
            result[size * i + j] /= result[size * j + j];
        }

        // Update the diagonal entry of this row.
        result[size*i+i] = A[size*i+i];
        for (int k = 0; k < i; ++k) {
            result[size * i + i] -= result[size * i + k] * result[size * i + k];
        }
        result[size * i + i] = sqrtf(result[size * i + i]);
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
    for (int i = n1; i < n2; ++i) {
        // Update the off diagonal entries.
        for (int j = 0; j < i; ++j) {
            result[size*i+j] = A[size*i+j];
            for (int k = 0; k < j; ++k) {
                result[size * i + j] -= result[size * i + k] * result[size * j + k];
            }
            result[size * i + j] /= result[size * j + j];
        }
        // Update the diagonal entry.
        result[size*i+i] = A[size*i+i];
        for (int k = 0; k < i; ++k) {
            result[size * i + i] -= result[size * i + k] * result[size * i + k];
        }
        result[size * i + i] = sqrtf(result[size * i + i]);
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
void incremental_cholesky_vect(double *A, double *result, int n1, int n2, int size) {
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


/*
 * Solver for a matrix that is in Cholesky decomposition.
 * Input arguments:
 *      d: dimension of matrix
 *      size: the actual size of the matrix
 *      LU: matrix
 *      b: right hand side
 *      x: matrix (d * 4) to put result in
 *      lower: if one the lower triangle system is solved, else the upper triangle system is solved.
*/
void cholesky_solve2_vect(int d, int size, double *LU, double *b0, double *b1, double *b2, double *b3, double *x, int lower) {
    __m256d vsum, vsum1, vsum2, vsum3, vsum4, v1, v2, v3, v4, v5;
    if (lower == 1) {
        for (int i = 0; i < d; ++i) {
            vsum1 = _mm256_setzero_pd();
            vsum2 = _mm256_setzero_pd();
            vsum3 = _mm256_setzero_pd();
            vsum4 = _mm256_setzero_pd();
            int k;
            for (k = 0; k + 3 < i; k+=4) {
                v1 = _mm256_set1_pd(LU[i * size + k]);
                v2 = _mm256_set1_pd(LU[i * size + k + 1]);
                v3 = _mm256_set1_pd(LU[i * size + k + 2]);
                v4 = _mm256_set1_pd(LU[i * size + k + 3]);
                v5 = _mm256_loadu_pd(&x[4 * k]);


                vsum1 = _mm256_fmadd_pd(v1, v5, vsum1);
                vsum2 = _mm256_fmadd_pd(v2, v5, vsum2);
                vsum3 = _mm256_fmadd_pd(v3, v5, vsum3);
                vsum4 = _mm256_fmadd_pd(v4, v5, vsum4);
            }
            vsum1 = _mm256_add_pd(vsum1, vsum2);
            vsum3 = _mm256_add_pd(vsum3, vsum4);
            vsum1 = _mm256_add_pd(vsum1, vsum3);
            while (k < i) {
                v1 = _mm256_set1_pd(LU[i * size + k]);
                v5 = _mm256_loadu_pd(&x[4 * k]);

                vsum1 = _mm256_fmadd_pd(v1, v5, vsum1);
                k += 1;
            }
            x[i * 4    ] = (b0[i] - vsum1[0]) / LU[i * size + i];
            x[i * 4 + 1] = (b1[i] - vsum1[1]) / LU[i * size + i];
            x[i * 4 + 2] = (b2[i] - vsum1[2]) / LU[i * size + i];
            x[i * 4 + 3] = (b3[i] - vsum1[3]) / LU[i * size + i];
        }
    } else {
        for (int i = d - 1; i >= 0; --i) {
            vsum = _mm256_setzero_pd();
            for (int k = i + 1; k < d; ++k) {
                v1 = _mm256_set1_pd(LU[i * size + k]);
                v2 = _mm256_loadu_pd(&x[4 * k]);
                vsum = _mm256_fmadd_pd(v1, v2, vsum);
            }
            x[i * 4    ] = (b0[i] - vsum[0]) / LU[i * size + i];
            x[i * 4 + 1] = (b1[i] - vsum[1]) / LU[i * size + i];
            x[i * 4 + 2] = (b2[i] - vsum[2]) / LU[i * size + i];
            x[i * 4 + 3] = (b3[i] - vsum[3]) / LU[i * size + i];
        }
    }
}


double frand() {
    return (double) rand() / (double) RAND_MAX * 100;
}


int main() {

    int n = 20;
    int gridsize = 100;


    double A[n * n];
    double result[n * n];
    double PSD[n * n];
    gsl_matrix *L = gsl_matrix_alloc(n, n);

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
    double X_grid[gridsize * gridsize * 2];
    double min = -5;
    double inc = 0.1;

    double x = min;
    for (int i = 0; i < gridsize; i++) {
        double y = min;
        for (int j = 0; j < gridsize; j++) {
            X_grid[i * 2 * gridsize + 2 * j] = y;
            X_grid[i * 2 * gridsize + 2 * j + 1] = x;
            y += inc;
        }
        x += inc;
    }


    // Initialize the sampled points:
    double X[n];
    for (double i=0; i<n; ++i) {
        int j = (int) i;
        X[j] = sin(i);
    }


    // Run the timing experiment for different grid sizes:
    for (int i = 20; i < gridsize; i+=4) {

        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                gsl_matrix_set(L, j, k, PSD[j * n + k]);
            }
        }


        clock_t start, end;
        double CPU_time, CPU_optimized;


        incremental_cholesky(PSD, result, 0, n, n);


        double v[n];
        double v1[n*4];

        start = clock();

        // Time cholesky decomposition:
//        for (int t=0; t<10000; t++) {
//            incremental_cholesky(PSD, result, 0, i, n);
//        }

        // Time system solve:
        for (int t=0; t<20; ++t) {
            for (int l = 0; l < i; l++) // for all points in X_grid
            {
                for (int j = 0; j < i; j++) // for all points in X_grid
                {
                    double x_star = X_grid[2 * gridsize * l + 2 * j]; // Current grid point that we are looking at
                    double y_star = X_grid[2 * gridsize * l + 2 * j + 1];
                    double k_star[n];

                    for (int k = 0; k < n; k++) {
                        int x = X[2 * k];
                        int y = X[2 * k + 1];
                        double arg1x = X_grid[x * 2 * gridsize + 2 * y];
                        double arg1y = X_grid[x * 2 * gridsize + 2 * y + 1];
                        k_star[k] = kernel2_baseline(&arg1x, &arg1y, &x_star, &y_star);
                    }

                    cholesky_solve2(n, n, result, k_star, v, 1);
                }
            }
        }

        end = clock();
        CPU_time = (end - start);

        start = clock();
        // Time cholesky decomposition:
//        for(int t=0; t<10000; t++) {
//            cholesky(PSD, result, i, n);
//        }

        // Time system solve
        for (int t=0; t<20; ++t) {
            for (int l = 0; l < i; l++) // for all points in X_grid
            {
                for (int j = 0; j < i; j += 4) // for all points in X_grid
                {
                    double x_star0, x_star1, x_star2, x_star3;
                    double y_star0, y_star1, y_star2, y_star3;
                    double k_star0[n], k_star1[n], k_star2[n], k_star3[n];

                    x_star0 = X_grid[2 * gridsize * l + 2 * (j)];
                    y_star0 = X_grid[2 * gridsize * l + 2 * (j) + 1];
                    x_star1 = X_grid[2 * gridsize * l + 2 * (j + 1)];
                    y_star1 = X_grid[2 * gridsize * l + 2 * (j + 1) + 1];
                    x_star2 = X_grid[2 * gridsize * l + 2 * (j + 2)];
                    y_star2 = X_grid[2 * gridsize * l + 2 * (j + 2) + 1];
                    x_star3 = X_grid[2 * gridsize * l + 2 * (j + 3)];
                    y_star3 = X_grid[2 * gridsize * l + 2 * (j + 3) + 1];

                    for (int kk = 0; kk < n; kk++) {
                        int x = X[2 * kk];
                        int y = X[2 * kk + 1];
                        double arg1x = X_grid[x * 2 * gridsize + 2 * y];
                        double arg1y = X_grid[x * 2 * gridsize + 2 * y + 1];
                        k_star0[kk] = kernel2_baseline(&arg1x, &arg1y, &x_star0, &y_star0);
                        k_star1[kk] = kernel2_baseline(&arg1x, &arg1y, &x_star1, &y_star1);
                        k_star2[kk] = kernel2_baseline(&arg1x, &arg1y, &x_star2, &y_star2);
                        k_star3[kk] = kernel2_baseline(&arg1x, &arg1y, &x_star3, &y_star3);
                    }
                    cholesky_solve2_vect(n, n, result, k_star0, k_star1, k_star2, k_star3, v1, 1);
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