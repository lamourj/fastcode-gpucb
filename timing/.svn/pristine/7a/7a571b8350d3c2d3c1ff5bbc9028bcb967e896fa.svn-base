#include "pow.h"

double pow_scalar (double x, uint32_t n) {
    if (n == 1) {
        return x;
    } else if (n == 0) {
        return 1.0;
    } else {
        int t = n / 2;
        int r = n % 2;
        double result = pow_scalar(x, t);
        if (r) {
            return result * result * x;
        } else {
            return result * result;
        }
    }
}