// Non-vector optimizations

#ifndef FASTCODE_GPUCB_GPUCB2_H
#define FASTCODE_GPUCB_GPUCB2_H

#include <stdbool.h>

double function(double x, double y);

void learn(double *X_grid, bool *sampled, int *X, double *T, int t, double *mu, double *sigma,
           double(*kernel)(double *, double *, double *, double *), double beta, int n);

double kernel2(double *x1, double *y1, double *x2, double *y2);

void initialize_meshgrid(double *X_grid, int n, double min, double inc);

void gpucb_initialized(int maxIter,
                       int n,
                       double *T,
                       int *X,
                       double *X_grid,
                       bool *sampled,
                       double *mu,
                       double *sigma,
                       double beta);

int gpucb(int maxIter, int n, double grid_min, double grid_inc);


#endif //FASTCODE_GPUCB_GPUCB2_H
