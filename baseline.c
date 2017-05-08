#include "stdio.h"
#include "math.h"
#include <stdio.h>
#include <gsl/gsl_linalg.h>
#include "stdio.h"

void cholesky(gsl_matrix_view X, int n) {
    /*
     * These functions factorize the symmetric, positive-definite square matrix A into the
     * Cholesky decomposition A = L L^T (or A = L L^H for the complex case). On input, the
     * values from the diagonal and lower-triangular part of the matrix A are used (the
     * upper triangular part is ignored). On output the diagonal and lower triangular part
     * of the input matrix A contain the matrix L, while the upper triangular part is
     * unmodified. If the matrix is not positive-definite then the decomposition will fail,
     *  returning the error code GSL_EDOM.
     */
    gsl_linalg_cholesky_decomp1(&X.matrix);
}

void solve(gsl_matrix_view A, gsl_vector_view b, int n, gsl_permutation *p, gsl_vector *x) {
    int s;
    gsl_linalg_LU_decomp(&A.matrix, p, &s);
    gsl_linalg_LU_solve(&A.matrix, p, &b.vector, x);
}


void cholesky_solve(gsl_matrix_view A, gsl_vector_view b, gsl_vector *x) {
    gsl_linalg_cholesky_solve(&A.matrix, &b.vector, x);
}


void gp_regression(double *X_grid, int *X, double *T, int t, double(*kernel)(double *, double *, double *, double *),
                   double *mu,
                   double *sigma, int n) {
    int t_gp = t + 1;
    // double L[t_gp * t_gp];
    double L_T[t_gp * t_gp];
    double K[t_gp * t_gp];

    // Build the K matrix
    for (int i = 0; i < t_gp; i++) {
        for (int j = 0; j < t_gp; j++) {
            int x1 = X[2 * i];
            int y1 = X[2 * i + 1];
            int x2 = X[2 * j];
            int y2 = X[2 * j + 1];

            K[i * t_gp + j] = (*kernel)(&X_grid[x1 * 2 * n + 2 * y1], &X_grid[x1 * 2 * n + 2 * y1 + 1],
                                        &X_grid[x2 * 2 * n + 2 * y2], &X_grid[x2 * 2 * n + 2 * y2 + 1]);
            // K is symmetric, shouldn't go through all entries when optimizing

            /*printf("t_gp: %d, x0: %lf, y0: %lf, x1: %lf, y1: %lf, k: %lf, ki:%d \n", t_gp, X_grid[x1 * 2 * n + 2 * y1],
                   X_grid[x1 * 2 * n + 2 * y1 + 1], X_grid[x2 * 2 * n + 2 * y2], X_grid[x2 * 2 * n + 2 * y2 + 1],
                   K[i * t_gp + j], i * t_gp + j);*/
        }
        //printf("\n");
    }

    // 2. Cholesky
    gsl_matrix_view K_view = gsl_matrix_view_array(K, t_gp, t_gp);
    cholesky(K_view, t_gp);
//    printf("Kernel matrix:\n");
//    for(int i = 0; i < t_gp; i++){
//        for(int j = 0; j < t_gp; j++){
//            printf("%lf ", K[i * t_gp + j]);
//        }
//        printf("\n");
//    }

//    for (int i = 0; i < t_gp - 1; i++) {
//        for (int j = i + 1; j < t_gp; j++) {
//            K[i * t_gp + j] = 0; //FIXME do we need that ? NO
//        }
//    }

//    for(int i = 0; i < t_gp; i++){
//        for(int j = 0; j < t_gp;j++){
//            printf("%lf ",K[i*t_gp+j]);
//        }
//        printf("\n");
//    }

    double *L = K;

    // 3. Compute alpha
    gsl_vector *x = gsl_vector_alloc(t_gp);
    gsl_vector *alpha = gsl_vector_alloc(t_gp);
    gsl_vector *v = gsl_vector_alloc(t_gp);
    gsl_permutation *p = gsl_permutation_alloc(t_gp);

    gsl_matrix_view L_view = gsl_matrix_view_array(L, t_gp, t_gp);
    gsl_matrix_view L_T_view = gsl_matrix_view_array(L_T, t_gp, t_gp);
    gsl_vector_view T_view = gsl_vector_view_array(T, t_gp);

    cholesky_solve(L_view, T_view, x);

    gsl_matrix_transpose_memcpy(&L_T_view.matrix, &L_view.matrix);

    gsl_vector_view x_view = gsl_vector_subvector(x, 0, t_gp);
    cholesky_solve(L_T_view, x_view, alpha);
    gsl_vector_free(x);

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
                f_star += k_star[k] * alpha->data[k];
            }

            mu[i * n + j] = f_star;
            //printf("write in mu at %d \n", i*n+j);
            gsl_vector_view k_star_view = gsl_vector_view_array(k_star, t_gp);
            cholesky_solve(L_view, k_star_view, v);
            //printf("loop solve done\n");

            double variance = (*kernel)(&x_star, &y_star, &x_star, &y_star);
            for (int k = 0; k < t_gp; k++) {
                variance -= v->data[k] * v->data[k];
            }

            sigma[i * n + j] = variance;

        }
    }
    gsl_vector_free(alpha);
    gsl_permutation_free(p);
    gsl_vector_free(v);
}