// This version includes the incremental cholesky factorization.

#ifndef FASTCODE_GPUCB_MATHHELPERS_H
#define FASTCODE_GPUCB_MATHHELPERS_H


void cholesky_baseline(double *A, int n, int size);

void incremental_cholesky_baseline(double *A, double *A_T, int n1, int n2, int size);

void cholesky_solve2_baseline(int d, int size, double *LU, double *b, double *x, int lower);

void cholesky_solve_baseline(int d, double *LU, double *b, double *x);

void transpose_baseline(double *M, double *M_T, int d, int size);

void gp_regression_baseline(double *X_grid,
                   double *K,
                   double *L_T,
                   int *X,
                   double *T,
                   int t,
                   int maxIter,
                   double   (*kernel)(double *, double *, double *, double *),
                   double *mu,
                   double *sigma,
                   int n);


#endif //FASTCODE_GPUCB_MATHHELPERS_H