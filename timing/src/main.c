#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "perf.h"
#include "test.h"

#define N 500


int main () {

    int i;
    uint64_t cycles_math_pow;

    perf_init ();

    // warm up the cache
    for (i = 0; i < N; i += 1) test();
    cycles_count_start();
    for (i = 0; i < N; i += 1) test();
    cycles_math_pow = cycles_count_stop();


    printf("math_pow  : %lf cycles\n", (double) cycles_math_pow / N);

    perf_done();

    return 0;
}