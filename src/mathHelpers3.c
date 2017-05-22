// Vectorized search for point to sample.

#include "mathHelpers3.h"
#include <math.h>


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
        A[size * i + i] = sqrt(A[size * i + i]);
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
void incremental_cholesky(double *A, double *A_T, int n1, int n2, int size) {
    for (int i = n1; i < n2; ++i) {
        // Update the off diagonal entries.
        for (int j = 0; j < i; ++j) {
            for (int k = 0; k < j; ++k) {
                A[size * i + j] -= A[size * i + k] * A[size * j + k];
            }
            A[size * i + j] /= A[size * j + j];
            A_T[size*j + i] = A[size * i + j];
        }
        // Update the diagonal entry.
        for (int k = 0; k < i; ++k) {
            A[size * i + i] -= A[size * i + k] * A[size * i + k];
        }
        A[size * i + i] = sqrtf(A[size * i + i]);
        A_T[size*i + i] = A[size * i + i];
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

// Old version.
void cholesky_solve(int d, double *LU, double *b, double *x) {
    double y[d];
    for (int i = 0; i < d; ++i) {
        double sum = 0.;
        for (int k = 0; k < i; ++k)sum += LU[i * d + k] * y[k];
        y[i] = (b[i] - sum) / LU[i * d + i];
    }
    for (int i = d - 1; i >= 0; --i) {
        double sum = 0.;
        for (int k = i + 1; k < d; ++k)sum += LU[k * d + i] * x[k];
        x[i] = (y[i] - sum) / LU[i * d + i];
    }
}


void transpose(double *M, double *M_T, int d, int size) {
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            M_T[j * size + i] = M[i * size + j];
        }
    }
}


void gp_regression(double *X_grid,
                   double *K,
                   double *L_T,
                   int *X,
                   double *T,
                   int t,
                   int maxIter,
                   double   (*kernel)(double *, double *, double *, double *),
                   double *mu,
                   double *sigma,
                   int n) {
    int t_gp = t + 1;

    // extend the K matrix
    int i = t_gp - 1;
    for (int j = 0; j < t_gp; j++) {
        int x1 = X[2 * i];
        int y1 = X[2 * i + 1];
        int x2 = X[2 * j];
        int y2 = X[2 * j + 1];

        K[i * maxIter + j] = (*kernel)(&X_grid[x1 * 2 * n + 2 * y1], &X_grid[x1 * 2 * n + 2 * y1 + 1],
                                       &X_grid[x2 * 2 * n + 2 * y2], &X_grid[x2 * 2 * n + 2 * y2 + 1]);
        // K is symmetric, shouldn't go through all entries when optimizing
        if(i==j){
            K[i * maxIter + j] += 0.5;
        }

        /*printf("t_gp: %d, x0: %lf, y0: %lf, x1: %lf, y1: %lf, k: %lf, ki:%d \n", t_gp, X_grid[x1 * 2 * n + 2 * y1],
               X_grid[x1 * 2 * n + 2 * y1 + 1], X_grid[x2 * 2 * n + 2 * y2], X_grid[x2 * 2 * n + 2 * y2 + 1],
               K[i * t_gp + j], i * t_gp + j);*/
    }


    // 2. Cholesky
    incremental_cholesky(K, L_T, t_gp - 1, t_gp, maxIter);

    // 3. Compute alpha
    double x[t_gp];
    double alpha[t_gp];
    double v[t_gp];


    cholesky_solve2(t_gp, maxIter, K, T, x, 1);
    cholesky_solve2(t_gp, maxIter, L_T, x, alpha, 0);

    // 4-6. For all points in grid, compute k*, mu, sigma

    for (int i = 0; i < n; i++) // for all points in X_grid
    {
        for (int j = 0; j < n; j++) // for all points in X_grid
        {
            double x_star = X_grid[2 * n * i + 2 * j]; // Current grid point that we are looking at
            double y_star = X_grid[2 * n * i + 2 * j + 1];
            double k_star[t_gp];

            for (int k = 0; k < t_gp; k++) {
                int x = X[2 * k];
                int y = X[2 * k + 1];
                double arg1x = X_grid[x * 2 * n + 2 * y];
                double arg1y = X_grid[x * 2 * n + 2 * y + 1];
                k_star[k] = (*kernel)(&arg1x, &arg1y, &x_star, &y_star);
            }

            double f_star = 0;
            for (int k = 0; k < t_gp; k++) {
                //f_star += k_star[k] * alpha->data[k];
                f_star += k_star[k] * alpha[k];
            }

            mu[i * n + j] = f_star;
            //printf("fstar is: %lf", f_star);
            //printf("write in mu at %d \n", i*n+j);
            cholesky_solve2(t_gp, maxIter, K, k_star, v, 1);
            //printf("loop solve done\n");

            double variance = (*kernel)(&x_star, &y_star, &x_star, &y_star);
            for (int k = 0; k < t_gp; k++) {
                //variance -= v->data[k] * v->data[k];
                variance -= v[k] * v[k];
            }


            if(variance < 0){
                variance = 0.0;
            }
            sigma[i * n + j] = variance;

        }
    }
}