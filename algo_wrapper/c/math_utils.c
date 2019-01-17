#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "math_utils.h"

double norm_l2(double *x, int x_len) {
    double result = 0.0;
    for (int i = 0; i < x_len; i++) {
        result += x[i] * x[i];
    }
    return sqrt(result);
}

void rand_shuffle(int *array, int n) {
    srand((unsigned int) time(NULL));
    int tmp;
    for (int i = n - 1; i > 0; i--) {
        int j = (int) (random() % (i + 1));
        tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}