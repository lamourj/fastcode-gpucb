// Baseline version.

#ifndef FASTCODE_GPUCB_MATHHELPERS_H
#define FASTCODE_GPUCB_MATHHELPERS_H


void cholesky_baseline(double *A, int n, int size);

void incremental_cholesky_baseline(float *A, int n1, int n2, int size);

void Crout_baseline(int d, double *S, double *D);

void solveCrout_baseline(int d, double *LU, double *b, double *x);

void solve_baseline(double *A, double *b, int n, double *x);

void cholesky_solve2_baseline(int d, double *LU, double *b, double *x, int lower);

void transpose_baseline(double *M, double *M_T, int d);

void gp_regression_baseline(double *X_grid,
                            int *X,
                            double *T,
                            int t,
                            double(*kernel)(double *, double *, double *, double *),
                            double *mu,
                            double *sigma,
                            int n);


#endif //FASTCODE_GPUCB_MATHHELPERS_H
