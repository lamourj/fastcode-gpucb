// Non-vector optimizations

#ifndef FASTCODE_GPUCB_MATHHELPERS3_H
#define FASTCODE_GPUCB_MATHHELPERS3_H


void cholesky(double *A, int n, int size);

void Crout(int d, double *S, double *D);

void solveCrout(int d, double *LU, double *b, double *x);

void solve(double *A, double *b, int n, double *x);

void cholesky_solve2(int d, double *LU, double *b, double *x, int lower);

void transpose(double *M, double *M_T, int d);

void gp_regression(double *X_grid,
                   int *X,
                   double *T,
                   int t,
                   double(*kernel)(double *, double *, double *, double *),
                   double *mu,
                   double *sigma,
                   int n);

#endif //FASTCODE_GPUCB_MATHHELPERS3_H
