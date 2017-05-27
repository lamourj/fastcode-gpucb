// Find maximum value (point to sample in next it.) while doing GP-regression update.

#ifndef FASTCODE_GPUCB_GPUCB7_H
#define FASTCODE_GPUCB_GPUCB7_H

#include <stdbool.h>
#include <immintrin.h>

float GRID_MIN_;
float GRID_INC_;
float BETA_;
extern const char *tag[10];
// Allocate memory
float *T_;
int *X_;
float *X_grid_;
bool *sampled_;
float *mu_;
float *sigma_;
float *K_;
float *L_;
int *maxIJ_;

int I_;
int N_;

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
           const float beta,
           int n,
           int *maxIJ);

float kernel2(float *x1, float *y1, float *x2, float *y2);

void initialize_meshgrid(float *X_grid, int n, float min, float inc);

void initialize(const int, const int);

void run();

void clean();

int gpucb(int maxIter, int n, float grid_min, float grid_inc);

void cholesky(float *A, int n, int size);

void incremental_cholesky(float *A, float *A_T, int n1, int n2, int size);

void cholesky_solve2(int d, int size, float *LU, float *b, float *x, int lower);

void mmm(int jj, int kk, int ll, int maxIter, int k_max, float *sums, float *K, float *v);

void mmm_vect(int jj, int kk, int ll, int maxIter, int k_max, float *sums, float *K, float *v);

void solve_triangle(float *X_grid, int *X, float *mu, float *sigma, float *alpha, int i, int jj, int kk, int ll, int n,
                    int maxIter, float *sums, float *K, float *v);

void cholesky_solve2_opt(int d, int size, float *LU, float *b, float *x, int lower);

void transpose(float *M, float *M_T, int d, int size);

void gp_regression_opt(float *X_grid,
                       float *K,
                       float *L_T,
                       int *X,
                       float *T,
                       int t,
                       int maxIter,
                       float   (*kernel)(float *, float *, float *, float *),
                       float *mu,
                       float *sigma,
                       bool *sampled,
                       float beta,
                       int n,
                       int *maxIJ);


#endif //FASTCODE_GPUCB_GPUCB7_H
