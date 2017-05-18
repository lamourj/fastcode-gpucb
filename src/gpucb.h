// Baseline version.

#ifndef FASTCODE_GPUCB_GPUCB_H
#define FASTCODE_GPUCB_GPUCB_H

#include <stdbool.h>

double function_baseline(double x, double y);

void learn_baseline(double *X_grid, bool *sampled, int *X, double *T, int t, double *mu, double *sigma,
           double(*kernel)(double *, double *, double *, double *), double beta, int n);

double kernel2_baseline(double *x1, double *y1, double *x2, double *y2);

void initialize_meshgrid_baseline(double *X_grid, int n, double min, double inc);

void gpucb_initialized_baseline(int maxIter, int n, double *T, int *X, double *X_grid, bool *sampled, double *mu, double *sigma,
                      double beta);

int gpucb_baseline(int maxIter, int n, double grid_min, double grid_inc);

#endif //FASTCODE_GPUCB_GPUCB_H
