// This version includes the incremental cholesky factorization.

#ifndef FASTCODE_BASELINE_H
#define FASTCODE_BASELINE_H

#include <stdbool.h>

float GRID_MIN_bl;
float GRID_INC_bl;
float BETA_bl;

// Allocate memory
float *T_bl;
int *X_bl;
float *X_grid_bl;
bool *sampled_bl;
float *mu_bl;
float *sigma_bl;
float *K_bl;
float *L_bl;

int I_bl;
int N_bl;

float function_baseline(float x, float y);

void learn_baseline(float *X_grid,
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
                    int n);

float kernel2_baseline(float *x1, float *y1, float *x2, float *y2);

void initialize_meshgrid_baseline(float *X_grid, int n, float min, float inc);

void initialize_bl(const int, const int);

void run_bl();

void clean_bl();

int gpucb_baseline(int maxIter, int n, float grid_min, float grid_inc);

void cholesky_baseline(float *A, int n, int size);

void incremental_cholesky_baseline(float *A, float *A_T, int n1, int n2, int size);

void cholesky_solve2_baseline(int d, int size, float *LU, float *b, float *x, int lower);

void cholesky_solve_baseline(int d, float *LU, float *b, float *x);

void transpose_baseline(float *M, float *M_T, int d, int size);

void gp_regression_baseline(float *X_grid,
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

#endif //FASTCODE_BASELINE_H
