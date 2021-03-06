//
// Created by Daan Nilis on 27/05/2017.
//

#ifndef FASTCODE_GPUCB_TRIANGLE_H
#define FASTCODE_GPUCB_TRIANGLE_H

#include <stdbool.h>
#include <immintrin.h>

float GRID_MIN_;
float GRID_INC_;
float BETA_;
extern const char *tag[20];
// Allocate memory
float *T_;
int *X_;
float *X_grid_;
bool *sampled_;
float *mu_;
float *sigma_;
float *K_;
float *L_;
float *l_;
float *alpha_;
float *v_;
float *sums_;
int *maxIJ_;

int I_;
int N_;

void initialize_meshgrid(float *X_grid, int n, float min, float inc);

void initialize(const int, const int);

void run();

void clean();

void solve_triangle(float *X_grid, int *X, float *mu, float *sigma, float *alpha, int i, int jj, int kk, int ll, int n,
                    int maxIter, float *sums, float *K, float *v);

void solve_triangle_vect(float *X_grid, int *X, float *mu, float *sigma, float *alpha, int i, int jj, int kk, int ll, int n,
                    int maxIter, int kMax, float *sums, float *K, float *v);

void incremental_cholesky(float *A, float *A_T, int n1, int n2, int size);

float frand();

#endif //FASTCODE_GPUCB_TRIANGLE_H
