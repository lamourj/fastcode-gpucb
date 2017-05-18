// This version includes the incremental cholesky factorization.

#ifndef FASTCODE_GPUCB_MATHHELPERS1_H
#define FASTCODE_GPUCB_MATHHELPERS1_H


void cholesky(double *A, int n, int size);

void incremental_cholesky(double *A, int n1, int n2, int size);

void cholesky_solve2(int d, double *LU, double *b, double *x, int lower);

void cholesky_solve(int d, double *LU, double *b, double *x);

void transpose(double *M, double *M_T, int d);

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
                   int n);


#endif //FASTCODE_GPUCB_MATHHELPERS1_H