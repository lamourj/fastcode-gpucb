//
// Created by Julien Lamour on 28.05.17.
//

#ifndef FASTCODE_GPUCB_GPUCB_FLOPS_H
#define FASTCODE_GPUCB_GPUCB_FLOPS_H

#include <stdbool.h>

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

int I_;
int N_;

unsigned long int _N_ADDS;
unsigned long int _N_MULS;
unsigned long int _N_DIVS;
unsigned long int _N_EXPS;
unsigned long int _N_SQRT;

float add(float a, float b);
float sub(float a, float b);
float divide(float a, float b);
float fexponential(float a);
float mult(float a, float b);
float square_root(float a);

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

void initialize(const int, const int);

void run();

void clean();

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

#endif //FASTCODE_GPUCB_GPUCB_FLOPS_H
