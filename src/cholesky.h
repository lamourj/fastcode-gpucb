#ifndef FASTCODE_GPUCB_GPUCB_H
#define FASTCODE_GPUCB_GPUCB_H

void initialize(const int, const int);

void run();

void clean();

float frand();

void incremental_cholesky(float *A, float *result, int n1, int n2, int size);

extern char* tag[50];


#endif //FASTCODE_GPUCB_GPUCB_H