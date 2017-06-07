#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "gpucb8.h" // CHANGE TO gpucb1.h for newer version

int main(int argc, char *argv[]) {
    // Execution variables
    const int N = 480; // Meshgrid size
    const int I = atoi(argv[1]); // GP-UCB # of iterations
//
    if (!(N % 8 == 0)) {
        fprintf(stderr, " n is not divisible by 4 !!!, n=%d", N);
    }
    if (I >= N) {
        fprintf(stderr, " WARNING: maxIter>=n: maxIter=%d, n=%d", I, N);
    }
    initialize(I, N);
    run();


    // Find maximum point:
    int maxI = 0;
    int maxJ = 0;
    double max = mu_[0];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double currentValue = mu_[i * N + j];
            if (currentValue > max) {
                max = currentValue;
                maxI = i;
                maxJ = j;
            }
        }
    }
    double maxX = X_grid_[maxI * 2 * N + 2 * maxJ];
    double maxY = X_grid_[maxI * 2 * N + 2 * maxJ + 1];
    printf("\nMaximal point found by C code is %f at [%f %f]\n\n", max, maxX, maxY);

    // Save output to file:
    if (true) {
        FILE *f = fopen("mu_c.txt", "w");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(f, "%lf, ", mu_[i * N + j]);
            }
            fprintf(f, "\n");
        }
        fclose(f);
    }
    clean();


    return 0;
}