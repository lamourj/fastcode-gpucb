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


#endif //FASTCODE_GPUCB_GPUCB3_H
