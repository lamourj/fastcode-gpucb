// Vectorized search for point to sample.

#include "gpucb3.h"
#include "mathHelpers3.h"
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <immintrin.h>

double function(double x, double y) {
    // double t = sin(x) + cos(y);
    double t = -pow(x, 2) - pow(y, 2);
    printf("(C code) Sampled: [%.2lf %.2lf] result %lf \n", x, y, t);
    return t;
}

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
                    double beta,
                    int n) {
    int maxI = 0;
    int maxJ = 0;
    int ourMaxI = 0;
    int ourMaxJ = 0;
    __m256d maxIs = _mm256_setzero_pd();
    __m256d maxJs = _mm256_setzero_pd();

    double firstMax = mu[0] + sqrt(beta) * sigma[0];
    int inj = 0;
    int i, j, zz;
    const int unrollingFactor = 4;
    __m256d max = _mm256_set1_pd(firstMax);
    __m256d sqrtBeta = _mm256_set1_pd(sqrt(beta));

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j += unrollingFactor) {
            __m256d mus = _mm256_loadu_pd(mu + inj);
            __m256d sigmas = _mm256_loadu_pd(sigma + inj);
            __m256d betaSigmas = _mm256_mul_pd(sqrtBeta, sigmas);
            __m256d currentValues = _mm256_add_pd(mus, betaSigmas);
            __m256d sampledValues = _mm256_set_pd(sampled[inj + 3], sampled[inj + 2], sampled[inj + 1], sampled[inj]);
            __m256d currentIndicesI = _mm256_set1_pd(i);
            __m256d currentIndicesJ = _mm256_set_pd(j + 3, j + 2, j + 1, j);

            __m256d compared = _mm256_cmp_pd(currentValues, max, 14); // 14 is _CMP_GT_OS
            __m256d comparedAndSampled = _mm256_andnot_pd(sampledValues, compared);

            max = _mm256_blendv_pd(max, currentValues, comparedAndSampled);
            maxIs = _mm256_blendv_pd(maxIs, currentIndicesI, comparedAndSampled);
            maxJs = _mm256_blendv_pd(maxJs, currentIndicesJ, comparedAndSampled);

            inj += 4;
        }
    }

    /*
    double maximums[4];
    double maxIValues[4];
    double maxJValues[4];

    _mm256_store_pd(maximums, max);
    _mm256_store_pd(maxIValues, maxIs);
    _mm256_store_pd(maxJValues, maxJs);

     ourMaxI = (int) maxIValues[0];
     ourMaxJ = (int) maxJValues[0];
     vectorMax = (int) maximums[0];

     for (zz = 3; zz >= 0; zz--) {
        if (max[zz] > vectorMax) {
            vectorMax = maximums[zz];
            ourMaxI = (int) maxIValues[zz];
            ourMaxJ = (int) maxJValues[zz];
        }
    }
     */

    ourMaxI = (int) maxIs[0];
    ourMaxJ = (int) maxJs[0];


    double vectorMax = max[0];

    for (zz = 3; zz >= 0; zz--) {
        if (max[zz] > vectorMax) {
            vectorMax = max[zz];
            ourMaxI = (int) maxIs[zz];
            ourMaxJ = (int) maxJs[zz];
        }
    }


    int v = (int) maxIs[zz];


    const int t2 = 2 * t;
    maxI = ourMaxI;
    maxJ = ourMaxJ;

    X[t2] = maxI;
    X[t2 + 1] = maxJ;
    const int maxInmaxJ = maxI * n + maxJ;
    sampled[maxInmaxJ] = true;
    const int maxInmaxJ2 = 2 * maxInmaxJ;
    T[t] = function_baseline(X_grid[maxInmaxJ2], X_grid[maxInmaxJ2 + 1]);
    gp_regression_baseline(X_grid, K, L_T, X, T, t, maxIter, kernel, mu, sigma,
                           n); // updating mu and sigma for every x in X_grid
}

double kernel2(double *x1, double *y1, double *x2, double *y2) {
    // RBF kernel
    double sigma = 1;
    return exp(-((*x1 - *x2) * (*x1 - *x2) + (*y1 - *y2) * (*y1 - *y2)) / (2 * sigma * sigma));
}

void initialize_meshgrid(double *X_grid, int n, double min, double inc) {
    double x = min;
    for (int i = 0; i < n; i++) {
        double y = min;
        for (int j = 0; j < n; j++) {
            X_grid[i * 2 * n + 2 * j] = y;
            X_grid[i * 2 * n + 2 * j + 1] = x; // With this assignment, meshgrid is the same as python code
            y += inc;
        }
        x += inc;
    }
}

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
                       int n) {
    for (int t = 0; t < maxIter; t++) {
        learn(X_grid, K, L_T, sampled, X, T, t, maxIter, mu, sigma, kernel2, beta, n);
    }
}

int gpucb(int maxIter, int n, double grid_min, double grid_inc) {

    // Allocations
    double mu[n * n];
    double sigma[n * n];
    double X_grid[2 * n * n];
    double K[maxIter * maxIter];
    double L_T[maxIter * maxIter];
    double T[maxIter];
    int X[2 * maxIter];
    bool sampled[n * n];
    const double beta = 100;

    // Initializations
    for (int i = 0; i < n * n; i++) {
        sampled[i] = false;
        mu[i] = 0;
        sigma[i] = 0.5;
    }
    initialize_meshgrid(X_grid, n, grid_min, grid_inc);


    // -------------------------------------------------------------
    //                  Done with initializations
    // -------------------------------------------------------------

    gpucb_initialized(X_grid, K, L_T, sampled, X, T, maxIter, mu, sigma, beta, n);

    // -------------------------------------------------------------
    //           Done with gpucb; rest is output writing
    // -------------------------------------------------------------

    FILE *f = fopen("mu_c.txt", "w");
    bool printMuConsole = false;
    bool printSigmaConsole = false;
    if (printMuConsole) {
        printf("Mu matrix after training: \n");
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(f, "%lf, ", mu[i * n + j]);
            if (printMuConsole) {
                printf("%.5lf ", mu[i * n + j]);
            }
        }
        fprintf(f, "\n");
        if (printMuConsole) {
            printf("\n");
        }
    }
    fclose(f);

    if (printSigmaConsole) {
        printf("\n\nSigma matrix after training: \n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%.5lf ", sigma[i * n + j]);
            }
            printf("\n");
        }
    }


    return 0;
}