#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "perf.h"
#include <string.h>
#include "baseline.h"


// ONLY CHANGE THIS
#include "../../src/gpucb5.h"
const int N_MIN = 200; // min meshgrid size
const int N_MAX = 200; // max meshgrid size. MUST BE REACHABLE WITH STEP
const int N_STEP = 100; // 
const int ITER_MIN = 5; // min number of iterations
const int ITER_MAX = 10; // max number of iterations. MUST BE REACHABLE WITH STEP
const int ITER_STEP = 5; // min number of iterations
const int NUM_RUNS = 1;
const int COMPARE_TO_BASELINE = 1; // 0 or 1

// DO NOT CHANGE THE FOLLOWING LINES

void validate(const int II, const int NN, double cycles_measured){
	printf("validate for I=%d, N=%d \n", II, NN);
	initialize_bl(II,NN);
	cycles_count_start();
	for (int j = 0; j < NUM_RUNS; j += 1) run_bl();
	uint64_t cycle_cnt = cycles_count_stop();
	double tot_mu_diff = 0.0;
	double tot_sigma_diff = 0.0;

	for(int ii=0; ii<II;ii++){
		double diff = mu_bl[ii] - mu_[ii];
		double abs = (diff < 0) ? -diff : diff;
		tot_mu_diff += abs;
	}

	for(int ii=0; ii<II;ii++){
		double diff = sigma_bl[ii] - sigma_[ii];
		double abs = (diff < 0) ? -diff : diff;
		tot_sigma_diff += abs;
	}
	if(tot_mu_diff > 0 || tot_sigma_diff > 0){
		printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		printf(" DOES NOT VALIDATE! difference mu: %lf, difference simga: %lf!\n", tot_mu_diff, tot_sigma_diff);
		printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	} else{
		printf("==============================================================\n");
		printf("VALIDATED: SPEEDUP: %lf                                  \n", (double) cycles_measured/ ((double)cycle_cnt / NUM_RUNS));
		printf("==============================================================\n");
	}
	clean();
	clean_bl();
}

int main() {
	if(!(N_STEP % 8 == 0) || !(N_MIN % 8 == 0)) {
        	printf("n or step is not divisible by 8 !!! \n");
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
		    	for (j = 0; j < NUM_RUNS; j += 1) run();
			clean();
	
			initialize(i,n);
			cycles_count_start();
			for (j = 0; j < NUM_RUNS; j += 1) run();
			cycle_cnt = cycles_count_stop();
			if(n < N_MAX || COMPARE_TO_BASELINE==0) clean();
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
		    	for (j = 0; j < NUM_RUNS; j += 1) run();
			clean();
	
			initialize(i,n);
			cycles_count_start();
			for (j = 0; j < NUM_RUNS; j += 1) run();
			cycle_cnt = cycles_count_stop();
			if(i < ITER_MAX || COMPARE_TO_BASELINE==0) clean();
			fprintf(fp, "%d\t%d\t%lf\n", n, i, (double) cycle_cnt / NUM_RUNS);
		}
		fclose(fp);
	}

	if(COMPARE_TO_BASELINE==1){
		if(ITER_MIN != ITER_MAX)
			validate(ITER_MAX, N_MIN, (double) cycle_cnt / NUM_RUNS);
		if(N_MIN != N_MAX)
			validate(ITER_MIN, N_MAX, (double) cycle_cnt / NUM_RUNS);
	}

	perf_done();

	

	    	
    return 0;
}


