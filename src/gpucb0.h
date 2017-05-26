// Baseline version without incremental cholesky.

#ifndef FASTCODE_GPUCB_GPUCB0_H
#define FASTCODE_GPUCB_GPUCB0_H
#include <stdbool.h>
float GRID_MIN_;
float GRID_INC_;
float BETA_;
extern const char *tag[30];
// Allocate memory
float *T_;
int *X_;
float *X_grid_;
bool *sampled_;
float *mu_;
float *sigma_;
float *K_;
float *L_;

int I_;
int N_;

float function_baseline(float x, float y);

void learn_baseline(float *X_grid,
                    bool *sampled,
                    int *X,
                    float *T,
                    int t,
                    float *mu,
                    float *sigma,
                    float(*kernel)(float *, float *, float *, float *),
                    float beta,
                    int n);

float kernel2_baseline(float *x1, float *y1, float *x2, float *y2);

void initialize_meshgrid_baseline(float *X_grid, int n, float min, float inc);

void gpucb_initialized_baseline(int maxIter,
                                int n,
                                float *T,
                                int *X,
                                float *X_grid,
                                bool *sampled,
                                float *mu,
                                float *sigma,
                                float beta);

int gpucb_baseline(int maxIter, int n, float grid_min, float grid_inc);

void initialize(const int, const int);
void run();
void clean();

void cholesky_baseline(float *A, float *A_T, int n, int size);

void incremental_cholesky_baseline(float *A, int n1, int n2, int size);

void Crout_baseline(int d, float *S, float *D);

void solveCrout_baseline(int d, float *LU, float *b, float *x);

void solve_baseline(float *A, float *b, int n, float *x);

void cholesky_solve2_baseline(int d, float *LU, float *b, float *x, int lower);

void transpose_baseline(float *M, float *M_T, int d);

void gp_regression_baseline(float *X_grid,
                            int *X,
                            float *T,
                            int t,
                            float(*kernel)(float *, float *, float *, float *),
                            float *mu,
                            float *sigma,
                            int n);

#endif //FASTCODE_GPUCB_GPUCB0_H
