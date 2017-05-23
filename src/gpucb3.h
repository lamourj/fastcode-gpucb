// Vectorized search for point to sample.

#ifndef FASTCODE_GPUCB_GPUCB3_H
#define FASTCODE_GPUCB_GPUCB3_H

#include <stdbool.h>

double function(double x, double y);

void learn(double *X_grid,
           double *K,
           double *L_T,
           bool *sampled,
           int *X,
           double *T,
           int t,
           int maxIter,
           double *mu,
           double *sigma,
           double(*kernel)(double *, double *, double *, double *),
           double beta,
           int n);

double kernel2(double *x1, double *y1, double *x2, double *y2);

void initialize_meshgrid(double *X_grid, int n, double min, double inc);

void gpucb_initialized(double *X_grid,
                       double *K,
                       double *L_T,
                       bool *sampled,
                       int *X,
                       double *T,
                       int maxIter,
                       double *mu,
                       double *sigma,
                       double beta,
                       int n);

int gpucb(int maxIter, int n, double grid_min, double grid_inc);


void cholesky(double *A, int n, int size);

void incremental_cholesky(double *A, double *A_T, int n1, int n2, int size);

void cholesky_solve2(int d, int size, double *LU, double *b, double *x, int lower);

void cholesky_solve(int d, double *LU, double *b, double *x);

void transpose(double *M, double *M_T, int d, int size);

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


#endif //FASTCODE_GPUCB_GPUCB3_H
