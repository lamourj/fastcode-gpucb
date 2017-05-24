// Vectorized search for point to sample.

#ifndef FASTCODE_GPUCB_GPUCB3_H
#define FASTCODE_GPUCB_GPUCB3_H

#include <stdbool.h>

float function(float x, float y);

void learn(float *X_grid,
           float *K,
           float *L_T,
           bool *sampled,
           int *X,
           float *T,
           int t,
           int maxIter,
           float *mu,
           float *sigma,
           float(*kernel)(float *, float *, float *, float *),
           float beta,
           int n);

float kernel2(float *x1, float *y1, float *x2, float *y2);

void initialize_meshgrid(float *X_grid, int n, float min, float inc);

void gpucb_initialized(float *X_grid,
                       float *K,
                       float *L_T,
                       bool *sampled,
                       int *X,
                       float *T,
                       int maxIter,
                       float *mu,
                       float *sigma,
                       float beta,
                       int n);

int gpucb(int maxIter, int n, float grid_min, float grid_inc);


void cholesky(float *A, int n, int size);

void incremental_cholesky(float *A, float *A_T, int n1, int n2, int size);

void cholesky_solve2(int d, int size, float *LU, float *b, float *x, int lower);

void cholesky_solve(int d, float *LU, float *b, float *x);

void transpose(float *M, float *M_T, int d, int size);

void gp_regression(float *X_grid,
                   float *K,
                   float *L_T,
                   int *X,
                   float *T,
                   int t,
                   int maxIter,
                   float   (*kernel)(float *, float *, float *, float *),
                   float *mu,
                   float *sigma,
                   int n);


#endif //FASTCODE_GPUCB_GPUCB3_H
