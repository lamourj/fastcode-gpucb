#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "perf.h"
#include <string.h>



// ONLY CHANGE THIS
#include "../../src/cholesky.h"
const int N_MIN = 200; // min meshgrid size
const int N_MAX = 200; // max meshgrid size
const int N_STEP = 100; // 
const int ITER_MIN = 5; // min number of iterations
const int ITER_MAX = 100; // min number of iterations
const int ITER_STEP = 5; // min number of iterations
const int NUM_RUNS = 3;

// DO NOT CHANGE THE FOLLOWING LINES
int main() {
	if(!(N_STEP % 4 == 0)) {
        	printf("n is not divisible by 4 !!! \n");
	}
	int j, n, i;
	uint64_t cycle_cnt;
	
	if( N_MIN != N_MAX){
		i = ITER_MIN;
		FILE *fp;
		char filename[50] = "";
		printf("the tag is:");
		printf("%s",*tag);
		strcpy(filename, "results/");
		strcat(filename, *tag);
		strcat(filename, "_N");
		strcat(filename, ".csv");

	   	fp = fopen(filename, "w+");
	   	fprintf(fp, "n\ti\tcycles\n");
		perf_init();
		for(n=N_MIN; n<=N_MAX; n+=N_STEP){
			initialize(i, n);
		    	// warm up the cache
		    	for (j = 0; i < NUM_RUNS; j += 1) run();
			clean();
	
			initialize(i,n);
			cycles_count_start();
			for (j = 0; j < NUM_RUNS; j += 1) run();
			cycle_cnt = cycles_count_stop();
			clean();
			fprintf(fp, "%d\t%d\t%lf\n", n, i, (double) cycle_cnt / NUM_RUNS);
		}
		fclose(fp);
	}
	
	printf("itermin: %d\n", ITER_MIN);
	if(ITER_MIN != ITER_MAX){
		printf("Comes here\n");
		n = N_MIN;
		FILE *fp;
		char filename[50];
		printf("the tag is:");
		printf("%s",*tag);
		strcpy(filename, "results/");
		strcat(filename, *tag);
		strcat(filename, "_I");
		strcat(filename, ".csv");

	   	fp = fopen(filename, "w+");
	   	fprintf(fp, "n\ti\tcycles\n");
		perf_init();
		for(i=ITER_MIN; i<=ITER_MAX; i+=ITER_STEP){
			initialize(i, n);
		    	// warm up the cache
		    	for (j = 0; i < NUM_RUNS; j += 1) run();
			clean();
	
			initialize(i,n);
			cycles_count_start();
			for (j = 0; j < NUM_RUNS; j += 1) run();
			cycle_cnt = cycles_count_stop();
			clean();
			fprintf(fp, "%d\t%d\t%lf\n", n, i, (double) cycle_cnt / NUM_RUNS);
		}
		fclose(fp);
	}
	perf_done();
	    	
    return 0;
}
