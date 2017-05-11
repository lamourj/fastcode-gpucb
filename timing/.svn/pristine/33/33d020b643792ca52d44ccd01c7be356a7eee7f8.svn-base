#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "perf.h"
#include "pow.h"
#include "pow_avx.h"
#include "pow_scalar.h"

#define N 1024

double   input          [N];
double   output_math    [N];
double   output_scalar  [N];
double   output_avx     [N];
uint32_t exponent       [N];


void d2h(double d, char * buf)
{
	if (sizeof(double) == sizeof(uint64_t)) {
		uint64_t vd;
		memcpy(&vd, &d, sizeof(uint64_t));
		sprintf(buf, "%016llx", vd);
	} else {
		fprintf(stderr, "sizeof(double) != sizeof(uint64_t)\n");
		fprintf(stderr, "Validation can not be performed on this architecture\n");
	}
}

void init_values () {

    int i = 0;
    uint32_t  e;
    double r0, r1, r2, range, base;

    srand(7919);

    input[0] = 5.17;      exponent[0] = 0;
    input[1] = 5.17;      exponent[1] = 1;
    input[2] = 5.17;      exponent[2] = 2;
    input[3] = 5.17;      exponent[2] = 3;
    input[4] = 5.17;      exponent[4] = 4;
    input[5] = 1.0000001; exponent[5] = INT32_MAX;

    for (i = 6; i < N; i += 1) {
        r0 = rand() % 2;
        r1 = ((double) rand()) / RAND_MAX;
        r2 = ((double) rand()) / RAND_MAX;
        if (r0) {
            e   = (uint32_t)(r1 * INT32_MAX);
        } else {
            e   = (uint32_t)(r1 * 1024);
        }
        range = pow(DBL_MAX, 1.0 / (double)e) - 1;
        base  = 1 + range * r2;
        input   [i] = base;
        exponent[i] = e;
    }
}

void validate () {
    int i;
    double t1, t2;
    char sb[128], st1[128], st2[128];
    for (i = 0; i < N; i += 1) {
        t1 = output_scalar[i];
        t2 = output_avx   [i];
        if (fabs(t1 - t2) / fabs(t1) > 0.001) {
            d2h(input[i], sb);
            d2h(t1, st1);
            d2h(t2, st2);
            fprintf(stderr, "Validation %d failed!\n", i);
            fprintf(stderr, "exponent   : %ld\n", (long) exponent[i]);
            fprintf(stderr, "base       : %s - %lf\n", sb, input[i]);
            fprintf(stderr, "pow_scalar : %s - %lf\n", st1, t1);
            fprintf(stderr, "pow_avx    : %s - %lf\n", st2, t2);
            exit(1);
        }
    }
    printf("\nValidated!\n");
}




int main () {

    int i;
    uint64_t cycles_math_pow;
    uint64_t cycles_avx;
    uint64_t cycles_scalar;

    pow_avx_init ();
    init_values ();
    perf_init ();

    // warm up the cache
    for (i = 0; i < N; i += 1) output_math[i] = pow(input[i], exponent[i]);
    cycles_count_start();
    for (i = 0; i < N; i += 1) output_math[i] = pow(input[i], exponent[i]);
    cycles_math_pow = cycles_count_stop();

    // warm up the cache
    for (i = 0; i < N; i += 1) output_scalar[i] = pow_scalar(input[i], exponent[i]);
    cycles_count_start();
    for (i = 0; i < N; i += 1) output_scalar[i] = pow_scalar(input[i], exponent[i]);
    cycles_scalar = cycles_count_stop();

    // warm up the cache
    for (i = 0; i < N; i += 1) output_avx[i] = pow_avx(input[i], exponent[i]);
    cycles_count_start();
    for (i = 0; i < N; i += 1) output_avx[i] = pow_avx(input[i], exponent[i]);
    cycles_avx = cycles_count_stop();

    printf("math_pow  : %lf cycles\n", (double) cycles_math_pow / N);
    printf("pow_scalar: %lf cycles\n", (double) cycles_scalar   / N);
    printf("pow_avx   : %lf cycles\n", (double) cycles_avx      / N);

    validate();

    perf_done();

    return 0;
}