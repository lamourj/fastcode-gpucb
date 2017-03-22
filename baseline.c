#include <stdio.h>
double variance;
double mean;

void update_L(){

}

void update_alpha(){

}

void update_f_star(){

}

void update_v(){

}

void solve_linear_system(double **A, double* b, double *x){

}

double get_variance(){
    return variance;
}

double get_mean(){
    return mean;
}

void gp_regression(double** X, double *y,  double(*covariance_func)(double*, double*),double noise, double sample){

}

void gp_ucb(){


}