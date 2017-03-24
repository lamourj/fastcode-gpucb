#include "baseline.c"

double function(double x) {
    return x;
}

void learn(double* X_grid, int* X, double* T, int t, double* mu, double* sigma, double(*kernel)(double*, double*), double beta, int n){
    /*
     * grid_idx = self.argmax_ucb()
    *  self.sample(self.X_grid[grid_idx])
    *  for every point x:
     *  gp_regression()
    *  gp.fit(self.X, self.T)
    *  mu1 = self.mu
     */

    int maxI = 0; int maxJ = 0;
    double max = mu[0] + sqrt(beta) * sigma[0];
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++) {
            double currentValue = mu[i * n + j] + beta * sigma[i * n + j];
            if (currentValue > max) {
                max = currentValue;
                maxI = i;
                maxJ = j;
            }
        }
    }

    X[2 * t] = maxI;
    X[2 * t + 1] = maxJ;

    T[t] = function(X_grid[maxI * n + maxJ]);
    gp_regression(X_grid, X, T, t, kernel, mu, sigma, n); // updating mu and sigma for every x in X_grid
}

double kernel2(double* x, double* y)
{
    return *x + *y;
}

int main(){
    printf("Welcome\n");
    // Define D, mu_0, sigma_0, kernel function k

    int maxIter = 20;
    double T[maxIter];
    int X[2 * maxIter];

    int n;
    double X_grid[n * n];
    double mu[n * n];
    double sigma[n * n];


    double beta;

    for(int t=0; t < maxIter; t++){
        learn(X_grid, X, T, t, mu, sigma, kernel2, beta, n);
    }

    return 0;
}
