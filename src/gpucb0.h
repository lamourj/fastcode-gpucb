// Baseline version without incremental cholesky.

#ifndef FASTCODE_GPUCB_GPUCB0_H
#define FASTCODE_GPUCB_GPUCB0_H
#include <stdbool.h>
double GRID_MIN_;
double GRID_INC_;
double BETA_;
extern const char *tag[30];
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
                    bool *sampled,
                    int *X,
                    double *T,
                    int t,
                    double *mu,
                    double *sigma,
                    double(*kernel)(double *, double *, double *, double *),
                    double beta,
                    int n);

double kernel2_baseline(double *x1, double *y1, double *x2, double *y2);

void initialize_meshgrid_baseline(double *X_grid, int n, double min, double inc);

void gpucb_initialized_baseline(int maxIter,
                                int n,
                                double *T,
                                int *X,
                                double *X_grid,
                                bool *sampled,
                                double *mu,
                                double *sigma,
                                double beta);

int gpucb_baseline(int maxIter, int n, double grid_min, double grid_inc);

void initialize(const int, const int);
void run();
void clean();

void cholesky_baseline(double *A, double *A_T, int n, int size);

void incremental_cholesky_baseline(float *A, int n1, int n2, int size);

void Crout_baseline(int d, double *S, double *D);

void solveCrout_baseline(int d, double *LU, double *b, double *x);

void solve_baseline(double *A, double *b, int n, double *x);

void cholesky_solve2_baseline(int d, double *LU, double *b, double *x, int lower);

void transpose_baseline(double *M, double *M_T, int d);

void gp_regression_baseline(double *X_grid,
                            int *X,
                            double *T,
                            int t,
                            double(*kernel)(double *, double *, double *, double *),
                            double *mu,
                            double *sigma,
                            int n);

#endif //FASTCODE_GPUCB_GPUCB0_H
