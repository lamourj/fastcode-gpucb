// Non-vector optimizations

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
    int i, j, k;

    for (i = 0; i < n; ++i) {
        // Update the off diagonal entries first.
        const int sizei = size * i;
        for (j = 0; j < i; ++j) {
            const int sizej = size * j;
            const int sizeij = sizei + j;

            double A_sizeij = A[sizeij];
            for (k = 0; k < j; ++k) {
                const double aikajk = A[sizei + k] * A[sizej + k];
                A_sizeij -= aikajk;
            }
            A_sizeij /= A[sizej + j];
            A[sizeij] = A_sizeij;
        }

        // Update the diagonal entry of this row.
        const int sizeii = sizei + i;
        double A_sizeii = A[sizeii];
        for (k = 0; k < i; ++k) {
            A_sizeii -= A[sizei + k] * A[sizei + k];
        }
        A_sizeii = sqrt(A_sizeii);
        A[sizeii] = A_sizeii;
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


/*
 * Solver for a matrix that is in Cholesky decomposition.
 * Input arguments:
 *      d: dimension of matrix
 *      LU: matrix
 *      b: right hand side
 *      x: vector to put result in
 *      lower: if one the lower triangle system is solved, else the upper triangle system is solved.
*/
void cholesky_solve2(int d, double *LU, double *b, double *x, int lower) {
    if (lower == 1) {
        for (int i = 0; i < d; ++i) {
            double sum = 0.;
            for (int k = 0; k < i; ++k) {
                sum += LU[i * d + k] * x[k];
            }
            const double bi = b[i];
            const double bi_sum = bi - sum;
            const double LUidi = LU[i * d + i];
            const double xi = bi_sum / LUidi;

            x[i] = xi;
        }
    } else {
        for (int i = d - 1; i >= 0; --i) {
            double sum = 0.;
            for (int k = i + 1; k < d; ++k) {
                sum += LU[k * d + i] * x[k];
            }
            const double bi = b[i];
            const double bi_sum = bi - sum;
            const double LUidi = LU[i * d + i];
            const double xi = bi_sum / LUidi;
            x[i] = xi;
        }
    }

}


void transpose(double *M, double *M_T, int d) {
    int i, j;
    for (i = 0; i < d; ++i) {
        for (j = 0; j < d; ++j) {
            M_T[j * d + i] = M[i * d + j];
        }
    }
}


void gp_regression(double *X_grid,
                   int *X,
                   double *T,
                   int t,
                   double(*kernel)(double *, double *, double *, double *),
                   double *mu,
                   double *sigma,
                   int n) {

    const int t_gp = t + 1;
    const int t_gp_squared = t_gp * t_gp;
    double L_T[t_gp_squared];
    double K[t_gp_squared];


    int i, j, j2;
    int i2 = 0;
    int itgpj = 0;
    const int n2 = 2 * n;
    const int jUnrolling = 4;
    const int jUnrolling_1 = jUnrolling - 1;
    const int tgpUnrolling = (t_gp / jUnrolling) * jUnrolling;
    // Build the K matrix

    // TODO: i, j blocking.
    if(t_gp >= jUnrolling) {
        for (i = 0; i < t_gp; i++) {
            j2 = 0;
            for (j = 0; j + jUnrolling_1 < t_gp; j+=jUnrolling) {
                // Unrolled by 4: allows vectorization easily.
                const int x10 = X[i2];
                const int y10 = X[i2 + 1];
                const int x20 = X[j2];
                const int y20 = X[j2 + 1];
                const int idx10 = x10 * n2 + 2 * y10;
                const int idx20 = x20 * n2 + 2 * y20;
                K[itgpj] = (*kernel)(&X_grid[idx10], &X_grid[idx10 + 1],
                                            &X_grid[idx20], &X_grid[idx20 + 1]);
                // K is symmetric, shouldn't go through all entries when optimizing
                j2 += 2;
                itgpj += 1;
                const int x11 = X[i2];
                const int y11 = X[i2 + 1];
                const int x21 = X[j2];
                const int y21 = X[j2 + 1];
                const int idx11 = x11 * n2 + 2 * y11;
                const int idx21 = x21 * n2 + 2 * y21;
                K[itgpj] = (*kernel)(&X_grid[idx11], &X_grid[idx11 + 1],
                                            &X_grid[idx21], &X_grid[idx21 + 1]);
                // K is symmetric, shouldn't go through all entries when optimizing
                j2 += 2;
                itgpj += 1;

                const int x12 = X[i2];
                const int y12 = X[i2 + 1];
                const int x22 = X[j2];
                const int y22 = X[j2 + 1];
                const int idx12 = x12 * n2 + 2 * y12;
                const int idx22 = x22 * n2 + 2 * y22;
                K[itgpj] = (*kernel)(&X_grid[idx12], &X_grid[idx12 + 1],
                                            &X_grid[idx22], &X_grid[idx22 + 1]);
                // K is symmetric, shouldn't go through all entries when optimizing
                j2 += 2;
                itgpj += 1;
                
                const int x13 = X[i2];
                const int y13 = X[i2 + 1];
                const int x23 = X[j2];
                const int y23 = X[j2 + 1];
                const int idx13 = x13 * n2 + 2 * y13;
                const int idx23 = x23 * n2 + 2 * y23;
                K[itgpj] = (*kernel)(&X_grid[idx13], &X_grid[idx13 + 1],
                                            &X_grid[idx23], &X_grid[idx23 + 1]);
                // K is symmetric, shouldn't go through all entries when optimizing
                j2 += 2;
                itgpj += 1;
            }
            for (j = tgpUnrolling; j < t_gp; j++) {
                const int x10 = X[i2];
                const int y10 = X[i2 + 1];
                const int x20 = X[j2];
                const int y20 = X[j2 + 1];
                const int idx10 = x10 * n2 + 2 * y10;
                const int idx20 = x20 * n2 + 2 * y20;
                K[itgpj] = (*kernel)(&X_grid[idx10], &X_grid[idx10 + 1],
                                            &X_grid[idx20], &X_grid[idx20 + 1]);
                // K is symmetric, shouldn't go through all entries when optimizing
                j2 += 2;
                itgpj += 1;
            }
            i2 += 2;
        }
    } else {
        for (i = 0; i < t_gp; i++) {
            j2 = 0;
            for (j = 0; j < t_gp; j++) {
                const int x10 = X[i2];
                const int y10 = X[i2 + 1];
                const int x20 = X[j2];
                const int y20 = X[j2 + 1];
                const int idx10 = x10 * n2 + 2 * y10;
                const int idx20 = x20 * n2 + 2 * y20;
                K[itgpj] = (*kernel)(&X_grid[idx10], &X_grid[idx10 + 1],
                                            &X_grid[idx20], &X_grid[idx20 + 1]);
                // K is symmetric, shouldn't go through all entries when optimizing
                j2 += 2;
                itgpj += 1;
            }
            i2 += 2;
        }
    }

    // 2. Cholesky
    cholesky(K, t_gp, t_gp);

    double *L = K;

    // 3. Compute alpha
    double x[t_gp];
    double alpha[t_gp];
    double v[t_gp];


    cholesky_solve2(t_gp, L, T, x, 1);

    transpose(L, L_T, t_gp); // TODO: Maybe do this more efficient
    cholesky_solve2(t_gp, L_T, x, alpha, 0);

    // 4-6. For all points in grid, compute k*, mu, sigma

    int inj = 0;
    for (i = 0; i < n; i++) // for all points in X_grid
    {
        for (j = 0; j < n; j++) // for all points in X_grid
        {
            // const int inj = in + j;
            const int idx = 2 * inj;
            double x_star = X_grid[idx]; // Current grid point that we are looking at
            double y_star = X_grid[idx + 1];
            double k_star[t_gp];
            
            for (int k = 0; k < t_gp; k++) {
                const int k2 = 2 * k;
                const int x = X[k2];
                const int y = X[k2 + 1];
                const int idx2 = x * n2 + 2 * y;
                const double arg1x = X_grid[idx2];
                const double arg1y = X_grid[idx2 + 1];
                k_star[k] = (*kernel)(&arg1x, &arg1y, &x_star, &y_star);
            }

            double f_star = 0;
            for (int k = 0; k < t_gp; k++) {
                //f_star += k_star[k] * alpha->data[k];
                f_star += k_star[k] * alpha[k];
            }

            mu[inj] = f_star;
            //printf("fstar is: %lf", f_star);
            //printf("write in mu at %d \n", i*n+j);
            cholesky_solve2(t_gp, L, k_star, v, 1);
            //printf("loop solve done\n");

            double variance = (*kernel)(&x_star, &y_star, &x_star, &y_star);
            for (int k = 0; k < t_gp; k++) {
                //variance -= v->data[k] * v->data[k];
                const double vk = v[k];
                const double vkvk = vk * vk;
                variance -= vkvk;
            }

            sigma[inj] = variance;
            inj += 1;
        }
    }
}