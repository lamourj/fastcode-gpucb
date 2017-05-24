// Vectorized search for point to sample.

#include "gpucb3.h"
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <immintrin.h>

float function(float x, float y) {
    // float t = sin(x) + cos(y);
    float t = -powf(x, 2) - powf(y, 2);
    // printf("(C code) Sampled: [%.2lf %.2lf] result %lf \n", x, y, t);
    return t;
}

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
                    float beta,
                    int n) {

    __m256d maxIs = _mm256_setzero_pd();
    __m256d maxJs = _mm256_setzero_pd();

    float firstMax = mu[0] + sqrtf(beta) * sigma[0];
    int inj = 0;
    int i, j, zz;
    const int unrollingFactor = 4;
    __m256d max = _mm256_set1_pd(firstMax);
    __m256d sqrtfBeta = _mm256_set1_pd(sqrtf(beta));

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j += unrollingFactor) {
            const __m256d mus = _mm256_loadu_pd(mu + inj);
            const __m256d sigmas = _mm256_loadu_pd(sigma + inj);
            const __m256d betaSigmas = _mm256_mul_pd(sqrtfBeta, sigmas);
            const __m256d currentValues = _mm256_add_pd(mus, betaSigmas);
            const __m256d sampledValues = _mm256_set_pd(sampled[inj + 3], sampled[inj + 2], sampled[inj + 1], sampled[inj]);
            const __m256d currentIndicesI = _mm256_set1_pd(i);
            const __m256d currentIndicesJ = _mm256_set_pd(j + 3, j + 2, j + 1, j);

            const __m256d compared = _mm256_cmp_pd(currentValues, max, 14); // 14 is _CMP_GT_OS
            const __m256d comparedAndSampled = _mm256_andnot_pd(sampledValues, compared);

            max = _mm256_blendv_pd(max, currentValues, comparedAndSampled);
            maxIs = _mm256_blendv_pd(maxIs, currentIndicesI, comparedAndSampled);
            maxJs = _mm256_blendv_pd(maxJs, currentIndicesJ, comparedAndSampled);

            inj += 4;
        }
    }

    int maxI = (int) maxIs[0];
    int maxJ = (int) maxJs[0];
    float vectorMax = max[0];

    for (zz = 3; zz >= 0; zz--) {
        if (max[zz] > vectorMax) {
            vectorMax = max[zz];
            maxI = (int) maxIs[zz];
            maxJ = (int) maxJs[zz];
        }
    }

    const int t2 = 2 * t;
    X[t2] = maxI;
    X[t2 + 1] = maxJ;
    const int maxInmaxJ = maxI * n + maxJ;
    sampled[maxInmaxJ] = true;
    const int maxInmaxJ2 = 2 * maxInmaxJ;
    T[t] = function(X_grid[maxInmaxJ2], X_grid[maxInmaxJ2 + 1]);
    gp_regression(X_grid, K, L_T, X, T, t, maxIter, kernel, mu, sigma,
                           n); // updating mu and sigma for every x in X_grid
}

float kernel2(float *x1, float *y1, float *x2, float *y2) {
    // RBF kernel
    float sigma = 1;
    
    return expf(-((*x1 - *x2) * (*x1 - *x2) + (*y1 - *y2) * (*y1 - *y2)) / (2 * sigma * sigma));
}

void initialize_meshgrid(float *X_grid, int n, float min, float inc) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            X_grid[i * 2 * n + 2 * j] = min + j * inc;
            X_grid[i * 2 * n + 2 * j + 1] =
                    min + i * inc; // With this assignment, meshgrid is the same as python code
        }
    }
}

void gpucb_initialized(float *X_grid,
                       float *K,
                       float *L_T,
                       bool *sampled,
                       int *X,
                       float *T,
                       int maxIter,
                       float *mu,
                       float *sigma,
                       float beta,
                       int n) {
    for (int t = 0; t < maxIter; t++) {
        learn(X_grid, K, L_T, sampled, X, T, t, maxIter, mu, sigma, kernel2, beta, n);
    }
}

int gpucb(int maxIter, int n, float grid_min, float grid_inc) {

    // Allocations
    float mu[n * n];
    float sigma[n * n];
    float X_grid[2 * n * n];
    float K[maxIter * maxIter];
    float L_T[maxIter * maxIter];
    float T[maxIter];
    int X[2 * maxIter];
    bool sampled[n * n];
    const float beta = 100;

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
            fprintf(f, "%f, ", mu[i * n + j]);
            if (printMuConsole) {
                printf("%.5f ", mu[i * n + j]);
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
                printf("%.5f ", sigma[i * n + j]);
            }
            printf("\n");
        }
    }


    return 0;
}



float function_baseline(float x, float y) {
    // float t = sin(x) + cos(y);
    float t = -powf(x, 2) - powf(y, 2);
    printf("(C code) Sampled: [%.2f %.2f] result %f \n", x, y, t);
    return t;
}

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
                    float beta,
                    int n) {
    /*
     * grid_idx = self.argmax_ucb()
    *  self.sample(self.X_grid[grid_idx])
    *  for every point x:
     *  gp_regression()
    *  gp.fit(self.X, self.T)
    *  mu1 = self.mu
     */
    bool debug = true;
    int maxI = 0;
    int maxJ = 0;
    float max = mu[0] + sqrtf(beta) * sigma[0];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float currentValue = mu[i * n + j] + sqrtf(beta) * sigma[i * n + j];

            if (!sampled[i * n + j] && (currentValue > max)) {
                max = currentValue;
                maxI = i;
                maxJ = j;
            }
        }
    }

    X[2 * t] = maxI;
    X[2 * t + 1] = maxJ;
    sampled[maxI * n + maxJ] = true;
    T[t] = function_baseline(X_grid[maxI * 2 * n + 2 * maxJ], X_grid[maxI * 2 * n + 2 * maxJ + 1]);
    gp_regression_baseline(X_grid, K, L_T, X, T, t, maxIter, kernel, mu, sigma, n); // updating mu and sigma for every x in X_grid
}

float kernel2_baseline(float *x1, float *y1, float *x2, float *y2) {
    // RBF kernel
    float sigma = 1;
    return expf(-((*x1 - *x2) * (*x1 - *x2) + (*y1 - *y2) * (*y1 - *y2)) / (2 * sigma * sigma));
}

void initialize_meshgrid_baseline(float *X_grid, int n, float min, float inc) {
    float x = min;
    for (int i = 0; i < n; i++) {
        float y = min;
        for (int j = 0; j < n; j++) {
            X_grid[i * 2 * n + 2 * j] = y;
            X_grid[i * 2 * n + 2 * j + 1] = x; // With this assignment, meshgrid is the same as python code
            y += inc;
        }
        x += inc;
    }
}

void gpucb_initialized_baseline(float *X_grid,
                                float *K,
                                float *L_T,
                                bool *sampled,
                                int *X,
                                float *T,
                                int maxIter,
                                float *mu,
                                float *sigma,
                                float beta,
                                int n) {
    for (int t = 0; t < maxIter; t++) {
        learn_baseline(X_grid, K, L_T, sampled, X, T, t, maxIter, mu, sigma, kernel2_baseline, beta, n);
    }
}

int gpucb_baseline(int maxIter, int n, float grid_min, float grid_inc) {

    // Allocations
    float       mu[n * n];
    float       sigma[n * n];
    float       X_grid[2 * n * n];
    float       K[maxIter * maxIter];
    float       L_T[maxIter * maxIter];
    float       T[maxIter];
    int          X[2 * maxIter];
    bool         sampled[n * n];
    const float beta = 100;

    // Initializations
    for (int i = 0; i < n * n; i++) {
        sampled[i] = false;
        mu[i] = 0;
        sigma[i] = 0.5;
    }
    initialize_meshgrid_baseline(X_grid, n, grid_min, grid_inc);


    // -------------------------------------------------------------
    //                  Done with initializations
    // -------------------------------------------------------------

    gpucb_initialized_baseline(X_grid, K, L_T, sampled, X, T, maxIter, mu, sigma, beta, n);

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

/*
 Straightforward implementation of inplace Cholesky decomposition of matrix A.
 Input arguments:
    A:    The matrix to decompose
    n:    The size of the data in matrix A to decompose
    size: The actual size of the rows
 */
void cholesky_baseline(float *A, int n, int size) {
    for (int i = 0; i < n; ++i) {

        // Update the off diagonal entries first.
        for (int j = 0; j < i; ++j) {
            for (int k = 0; k < j; ++k) {
                A[size * i + j] -= A[size * i + k] * A[size * j + k];
            }
            A[size * i + j] /= A[size * j + j];
        }

        // Update the diagonal entry of this row.
        for (int k = 0; k < i; ++k) {
            A[size * i + i] -= A[size * i + k] * A[size * i + k];
        }
        A[size * i + i] = sqrtf(A[size * i + i]);
    }
}

/*
 Incremental implementation of Cholesky decomposition:
 The matrix contains a Cholesky decomposition until row n1,
 rows n1, to n2 are new data.
 Input arguments:
    A:    Partially decomposed matrix with new data from row n1, to n2
    n1:   Start of the new data
    n2:   End of the new data
    size: The actual size of the rows

 */
void incremental_cholesky_baseline(float *A, float *A_T, int n1, int n2, int size) {
    for (int i = n1; i < n2; ++i) {
        // Update the off diagonal entries.
        for (int j = 0; j < i; ++j) {
            for (int k = 0; k < j; ++k) {
                A[size * i + j] -= A[size * i + k] * A[size * j + k];
            }
            A[size * i + j] /= A[size * j + j];
            A_T[size*j + i] = A[size * i + j];
        }
        // Update the diagonal entry.
        for (int k = 0; k < i; ++k) {
            A[size * i + i] -= A[size * i + k] * A[size * i + k];
        }
        A[size * i + i] = sqrtff(A[size * i + i]);
        A_T[size*i + i] = A[size * i + i];
    }

}


/*
 * Solver for a matrix that is in Cholesky decomposition.
 * Input arguments:
 *      d: dimension of matrix
 *      size: the actual size of the matrix
 *      LU: matrix
 *      b: right hand side
 *      x: vector to put result in
 *      lower: if one the lower triangle system is solved, else the upper triangle system is solved.
*/
void cholesky_solve2_baseline(int d, int size, float *LU, float *b, float *x, int lower) {
    if (lower == 1) {
        for (int i = 0; i < d; ++i) {
            float sum = 0.;
            for (int k = 0; k < i; ++k) {
                sum += LU[i * size + k] * x[k];
            }
            x[i] = (b[i] - sum) / LU[i * size + i];
        }
    } else {
        for (int i = d - 1; i >= 0; --i) {
            float sum = 0.;
            for (int k = i + 1; k < d; ++k) {
                sum += LU[i * size + k] * x[k];
            }
            x[i] = (b[i] - sum) / LU[i * size + i];
        }
    }

}

// Old version.
void cholesky_solve_baseline(int d, float *LU, float *b, float *x) {
    float y[d];
    for (int i = 0; i < d; ++i) {
        float sum = 0.;
        for (int k = 0; k < i; ++k)sum += LU[i * d + k] * y[k];
        y[i] = (b[i] - sum) / LU[i * d + i];
    }
    for (int i = d - 1; i >= 0; --i) {
        float sum = 0.;
        for (int k = i + 1; k < d; ++k)sum += LU[k * d + i] * x[k];
        x[i] = (y[i] - sum) / LU[i * d + i];
    }
}


void transpose_baseline(float *M, float *M_T, int d, int size) {
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            M_T[j * size + i] = M[i * size + j];
        }
    }
}


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
                            int n) {
    int t_gp = t + 1;

    // extend the K matrix
    int i = t_gp - 1;
    for (int j = 0; j < t_gp; j++) {
        int x1 = X[2 * i];
        int y1 = X[2 * i + 1];
        int x2 = X[2 * j];
        int y2 = X[2 * j + 1];

        K[i * maxIter + j] = (*kernel)(&X_grid[x1 * 2 * n + 2 * y1], &X_grid[x1 * 2 * n + 2 * y1 + 1],
                                       &X_grid[x2 * 2 * n + 2 * y2], &X_grid[x2 * 2 * n + 2 * y2 + 1]);
        // K is symmetric, shouldn't go through all entries when optimizing
        if(i==j){
            K[i * maxIter + j] += 0.5;
        }
    }


    // 2. Cholesky
    incremental_cholesky_baseline(K, L_T, t_gp - 1, t_gp, maxIter);

    // 3. Compute alpha
    float x[t_gp];
    float alpha[t_gp];
    float v[t_gp];


    cholesky_solve2_baseline(t_gp, maxIter, K, T, x, 1);
    cholesky_solve2_baseline(t_gp, maxIter, L_T, x, alpha, 0);

    // 4-6. For all points in grid, compute k*, mu, sigma

    for (int i = 0; i < n; i++) // for all points in X_grid ([i])
    {
        for (int j = 0; j < n; j++) // for all points in X_grid ([i][j])
        {
            float x_star = X_grid[2 * n * i + 2 * j]; // Current grid point that we are looking at
            float y_star = X_grid[2 * n * i + 2 * j + 1];
            float k_star[t_gp];

            for (int k = 0; k < t_gp; k++) {
                int x = X[2 * k];
                int y = X[2 * k + 1];
                float arg1x = X_grid[x * 2 * n + 2 * y];
                float arg1y = X_grid[x * 2 * n + 2 * y + 1];
                k_star[k] = (*kernel)(&arg1x, &arg1y, &x_star, &y_star);
            }

            float f_star = 0;
            for (int k = 0; k < t_gp; k++) {
                //f_star += k_star[k] * alpha->data[k];
                f_star += k_star[k] * alpha[k];
            }

            mu[i * n + j] = f_star;
            //printf("fstar is: %lf", f_star);
            //printf("write in mu at %d \n", i*n+j);
            cholesky_solve2_baseline(t_gp, maxIter, K, k_star, v, 1);
            //printf("loop solve done\n");

            float variance = (*kernel)(&x_star, &y_star, &x_star, &y_star);
            for (int k = 0; k < t_gp; k++) {
                //variance -= v->data[k] * v->data[k];
                variance -= v[k] * v[k];
            }


            if(variance < 0){
                variance = 0.0;
            }
            sigma[i * n + j] = variance;

        }
    }
}