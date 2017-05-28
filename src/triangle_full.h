// Timing the triangle solve over the complete grid.
#ifndef FASTCODE_GPUCB_TRIANGLE_FULL_H
#define FASTCODE_GPUCB_TRIANGLE_FULL_H


#include <stdbool.h>
#include <immintrin.h>

float GRID_MIN_;
float GRID_INC_;
extern const char *tag[20];
// Allocate memory
float *X_grid_;
int *X_;
float *mu_;
float *sigma_;
float *K_;
float *x;
float *v;
float *k_star;
float *alpha;
float *sums;
float *K_;
float *A;
int I_;
int N_;

float frand();
void incremental_cholesky(float *A, float *A_T, int n1, int n2, int size);
void initialize_meshgrid(float *X_grid, int n, float min, float inc);

void initialize(const int, const int);

void run();

void clean();

void mmm(int jj, int kk, int ll, int maxIter, int k_max, float *sums, float *K, float *v);

void mmm_vect(int jj, int kk, int ll, int maxIter, int k_max, float *sums, float *K, float *v);

void solve_triangle(float *X_grid, int *X, float *mu, float *sigma, float *alpha, int i, int jj, int kk, int ll, int n,
                    int maxIter, int k_max, float *sums, float *K, float *v);

void
solve_triangle_vect(float *X_grid, int *X, float *mu, float *sigma, float *alpha, int i, int jj, int kk, int ll, int n,
                    int maxIter, int k_max, float *sums, float *K, float *v, float* k_star);



#endif //FASTCODE_GPUCB_TRIANGLE_FULL_H
