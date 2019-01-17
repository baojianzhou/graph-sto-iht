//
// Created by baojian on 8/16/18.
//

#include <cblas.h>
#include <stdio.h>
#include <stdbool.h>
#include "loss.h"

bool test_matrix_vector() {
    int n = 3, p = 4;
    double x_tr[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    double w[] = {1., 1., 1., 1.};
    double yz[] = {1., 2., 3.};
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                n, p, 1., x_tr, p, w, 1, 1., yz, 1);
    printf("matrix * vector: %lf %lf %lf\n", yz[0], yz[1], yz[2]);
    return true;
}

bool test_log_sum_exp() {
    double x[] = {1., 0., -1.};
    int x_len = 3;
    printf("log_sum_exp of x: %lf\n", log_sum_exp(x, x_len));
    return true;
}

bool test_logistic() {
    double x[] = {1., 0., -1.}, out[3];
    int x_len = 3;
    logistic(x, out, x_len);
    printf("log_sum_exp of x: %lf %lf %lf\n", out[0], out[1], out[2]);
    return true;
}

bool test_least_square() {
    printf("--------------------\n");
    int n_samples = 3, n_features = 4;
    double x_tr[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    double w[] = {1., 1., 1., 1.};
    double yz[] = {1., 2., 3., 4.};
    double y_tr[] = {1., 2., 3.};
    double loss_grad[] = {1., 2., 3., 1., 2., 3.};
    double eta = 0.0;
    cblas_dgemv(CblasRowMajor, CblasTrans,
                n_samples, n_features, 1., x_tr, n_features, w, 1, 1., yz, 1);
    printf("matrix * vector: %lf %lf %lf %lf\n", yz[0], yz[1], yz[2], yz[3]);
    ls_loss_grad(w, x_tr, y_tr, loss_grad, eta, n_samples,
                 n_features);
    return true;
}

int main() {
    double x1 = (10 - 1.) / 10;
    double x2 = (10 - 1.) / 10;
    printf("%f %f", x1, x2);
    test_matrix_vector();
    test_log_sum_exp();
    test_logistic();
    test_least_square();
}