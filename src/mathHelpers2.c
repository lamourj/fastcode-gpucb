// Non-vector optimizations

#include "mathHelpers2.h"
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


    int i, j;
    // Build the K matrix
    for (i = 0; i < t_gp; i++) {
        for (j = 0; j < t_gp; j++) {
            const int x1 = X[2 * i];
            const int y1 = X[2 * i + 1];
            const int x2 = X[2 * j];
            const int y2 = X[2 * j + 1];

            K[i * t_gp + j] = (*kernel)(&X_grid[x1 * 2 * n + 2 * y1], &X_grid[x1 * 2 * n + 2 * y1 + 1],
                                        &X_grid[x2 * 2 * n + 2 * y2], &X_grid[x2 * 2 * n + 2 * y2 + 1]);
            // K is symmetric, shouldn't go through all entries when optimizing
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

    for (i = 0; i < n; i++) // for all points in X_grid
    {
        for (j = 0; j < n; j++) // for all points in X_grid
        {
            const int inj = i * n + j;
            const int idx = 2 * inj;
            double x_star = X_grid[idx]; // Current grid point that we are looking at
            double y_star = X_grid[idx + 1];
            double k_star[t_gp];

            for (int k = 0; k < t_gp; k++) {
                const int k2 = 2 * k;
                const int x = X[k2];
                const int y = X[k2 + 1];
                const int idx2 = x * 2 * n + 2 * y;
                const double arg1x = X_grid[idx2];
                const double arg1y = X_grid[idx2 + 1];
                k_star[k] = (*kernel)(&arg1x, &arg1y, &x_star, &y_star);
            }

            double f_star = 0;
            for (int k = 0; k < t_gp; k++) {
                //f_star += k_star[k] * alpha->data[k];
                f_star += k_star[k] * alpha[k];
            }
            /*
            if(f_star != f_star) {
                printf("Nan problem fstar\n");
            }
            if(f_star >= 200) {
                printf("f_star: %lf , t: %d\n", f_star, t);
            }*/
            /*
            if(t >= 60) {
                printf("%lf\n", f_star);
            }*/
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

            /*
            if (variance < 0) {
                variance = 0.0;
            }*/

            sigma[inj] = variance;

        }
    }
}