// Baseline version.

#include "gpucb.h"
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

void learn(double *X_grid, bool *sampled, int *X, double *T, int t, double *mu, double *sigma,
           double(*kernel)(double *, double *, double *, double *), double beta, int n) {
    /*
     * grid_idx = self.argmax_ucb()
    *  self.sample(self.X_grid[grid_idx])
    *  for every point x:
     *  gp_regression()
    *  gp.fit(self.X, self.T)
    *  mu1 = self.mu
     */
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

            /*
            double currentValue0 = currentValues[0];
            if (currentValue0 > firstMax &&
                !sampled[inj]) { // Maybe faster if inverted, especially after blocking? Lookup faster than comparison.
                firstMax = currentValue0;
                maxI = i;
                maxJ = j;
            }
            inj++;
            double currentValue1 = currentValues[1];
            if (currentValue1 > firstMax &&
                !sampled[inj]) { // Maybe faster if inverted, especially after blocking? Lookup faster than comparison.
                firstMax = currentValue1;
                maxI = i;
                maxJ = j + 1;
            }
            inj++;
            double currentValue2 = currentValues[2];
            if (currentValue2 > firstMax &&
                !sampled[inj]) { // Maybe faster if inverted, especially after blocking? Lookup faster than comparison.
                firstMax = currentValue2;
                maxI = i;
                maxJ = j + 2;
            }
            inj++;
            double currentValue3 = currentValues[3];
            if (currentValue3 > firstMax &&
                !sampled[inj]) { // Maybe faster if inverted, especially after blocking? Lookup faster than comparison.
                firstMax = currentValue3;
                maxI = i;
                maxJ = j + 3;
            }
            inj++;
            */

            inj += 4;
            /*
            double maximums[4];
            double maxIValues[4];
            double maxJValues[4];

            _mm256_store_pd(maximums, max);
            _mm256_store_pd(maxIValues, maxIs);
            _mm256_store_pd(maxJValues, maxJs);

            ourMaxI = (int) maxIValues[0];
            ourMaxJ = (int) maxJValues[0];


            double vectorMax = maximums[0];

            for (zz = 3; zz >=0; zz--) {
                if (maximums[zz] > vectorMax) {
                    vectorMax = maximums[zz];
                    ourMaxI = (int) maxIValues[zz];
                    ourMaxJ = (int) maxJValues[zz];
                }
            }*/


            /*
            if(vectorMax >= firstMax) {

            } else {
                printf("Problem value %lf \n", vectorMax - firstMax);
            }

            if(ourMaxI == maxI && ourMaxJ == maxJ) {

            } else {
                /*
                printf("Problem indices for i, j %d %d \n", i, j);

                printf("max0: %lf  max1: %lf  max2: %lf  max3: %lf\n", maximums[0], maximums[1], maximums[2], maximums[3]);
                printf("maxi: %d  maxj: %d \n", maxI, maxJ);
                printf("ourI: %d  ourJ: %d\n", ourMaxI, ourMaxJ);
                printf("i0: %d  i1: %d  i2: %d  i3: %d\n", (int)maxIValues[0], (int)maxIValues[1], (int)maxIValues[2], (int)maxIValues[3]);
                printf("j0: %d  j1: %d  j2: %d  j3: %d\n", (int)maxJValues[0], (int)maxJValues[1], (int)maxJValues[2], (int)maxJValues[3]);

            }

            if(false && currentValue0 == currentValue3) {
                printf("Maybe understood\n");
            }*/
        }
    }
    double maximums[4];
    double maxIValues[4];
    double maxJValues[4];

    _mm256_store_pd(maximums, max);
    _mm256_store_pd(maxIValues, maxIs);
    _mm256_store_pd(maxJValues, maxJs);

    ourMaxI = (int) maxIValues[0];
    ourMaxJ = (int) maxJValues[0];


    double vectorMax = maximums[0];

    for (zz = 3; zz >=0; zz--) {
        if (maximums[zz] > vectorMax) {
            vectorMax = maximums[zz];
            ourMaxI = (int) maxIValues[zz];
            ourMaxJ = (int) maxJValues[zz];
        }
    }



    const int t2 = 2 * t;
    maxI = ourMaxI;
    maxJ = ourMaxJ;
    X[t2] = maxI;
    X[t2 + 1] = maxJ;
    const int maxInmaxJ = maxI * n + maxJ;
    sampled[maxInmaxJ] = true;
    const int maxInmaxJ2 = 2 * maxInmaxJ;
    T[t] = function(X_grid[maxInmaxJ2], X_grid[maxInmaxJ2 + 1]);
    gp_regression(X_grid, X, T, t, kernel, mu, sigma, n); // updating mu and sigma for every x in X_grid
    // printf("\n");
}

double kernel2(double *x1, double *y1, double *x2, double *y2) {
    // RBF kernel
    double sigma = 1;
    const double x1_x2 = *x1 - *x2;
    const double y1_y2 = *y1 - *y2;
    const double x1212 = x1_x2 * x1_x2;
    const double y1212 = y1_y2 * y1_y2;
    const double numerator = x1212 + y1212;
    const double arg = numerator / 2;
    const double negArg = -arg;

    // return exp(-(x1212 + y1212) / (2 * sigma * sigma));
    return exp(negArg);
}

// Not timed... doesn't have to be optimized at the moment.
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

void gpucb_initialized(int maxIter,
                       int n,
                       double *T,
                       int *X,
                       double *X_grid,
                       bool *sampled,
                       double *mu,
                       double *sigma,
                       double beta) {
    for (int t = 0; t < maxIter; t++) {
        learn(X_grid, sampled, X, T, t, mu, sigma, kernel2, beta, n);
    }
}

int gpucb(int maxIter, int n, double grid_min, double grid_inc) {

    // Allocations
    double T[maxIter];
    int X[2 * maxIter];
    double X_grid[2 * n * n];
    bool sampled[n * n];
    double mu[n * n];
    double sigma[n * n];
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

    gpucb_initialized(maxIter, n, T, X, X_grid, sampled, mu, sigma, beta);

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