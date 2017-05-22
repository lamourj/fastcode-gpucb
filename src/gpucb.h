// This version includes the incremental cholesky factorization.

#ifndef FASTCODE_GPUCB_GPUCB_H
#define FASTCODE_GPUCB_GPUCB_H
#include <stdbool.h>

double GRID_MIN_;
double GRID_INC_;
double BETA_;
extern const char *tag[10];
// Allocate memory
double *T_;
int *X_;
double *X_grid_;
bool *sampled_;
double *mu_;
double *sigma_;
double *K_;
double *L_;

int I_;
int N_;

double function_baseline(double x, double y);

void learn_baseline(double *X_grid,
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
                    const double beta,
                    int n);

double kernel2_baseline(double *x1, double *y1, double *x2, double *y2);

void initialize_meshgrid_baseline(double *X_grid, int n, double min, double inc);

void initialize(const int, const int);
void run();
void clean();

int gpucb_baseline(int maxIter, int n, double grid_min, double grid_inc);

void cholesky_baseline(double *A, int n, int size);

void incremental_cholesky_baseline(double *A, double *A_T, int n1, int n2, int size);

void cholesky_solve2_baseline(int d, int size, double *LU, double *b, double *x, int lower);

void cholesky_solve_baseline(int d, double *LU, double *b, double *x);

void transpose_baseline(double *M, double *M_T, int d, int size);

void gp_regression_baseline(double *X_grid,
                            double *K,
                            double *L_T,
                            int *X,
                            double *T,
                            int t,
                            int maxIter,
                            double   (*kernel)(double *, double *, double *, double *),
                            double *mu,
                            double *sigma,
                            int n);

#endif //FASTCODE_GPUCB_GPUCB_H
