//
// Created by Julien Lamour on 26.05.17.
//

#ifndef FASTCODE_GPUCB_MATRIX_H
#define FASTCODE_GPUCB_MATRIX_H

#include <immintrin.h>

extern const char *tag[10];

void initialize(const int, const int);

void run();

void clean();

float frand();

void mmm_vect(int jj, int kk, int ll, int maxIter, int t_gp, float *sums, float *K, float *v);

float hsum_mm256(__m256 x);

void mmm(int jj, int kk, int ll, int maxIter, int t_gp, float *sums, float *K, float *v);

#endif //FASTCODE_GPUCB_MATRIX_H
