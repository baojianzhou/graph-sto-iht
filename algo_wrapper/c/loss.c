//
// Created by baojian on 8/16/18.
//
#include <cblas.h>
#include <stdlib.h>
#include <string.h>
#include "loss.h"

void expit(const double *x, double *out, int x_len) {
    for (int i = 0; i < x_len; i++) {
        if (x[i] > 0) {
            out[i] = 1. / (1. + exp(-x[i]));
        } else {
            out[i] = 1. - 1. / (1. + exp(x[i]));
        }
    }
}


void log_logistic(const double *x, double *out, int x_len) {
    for (int i = 0; i < x_len; i++) {
        if (x[i] > 0.0) {
            out[i] = -log(1.0 + exp(-x[i]));
        } else {
            out[i] = x[i] - log(1.0 + exp(x[i]));
        }
    }
}

void logistic(const double *x, double *out, int x_len) {
    for (int i = 0; i < x_len; i++) {
        if (x[i] > 0) {
            out[i] = 1. / (1. + exp(-x[i]));
        } else {
            out[i] = 1. - 1. / (1. + exp(x[i]));
        }
    }
}

double log_sum_exp(const double *x, int x_len) {
    double max_x = x[0], out = 0.0;
    for (int i = 1; i < x_len; i++) {
        if (x[i] > max_x) {
            max_x = x[i];
        }
    }
    for (int i = 0; i < x_len; i++) {
        out += exp(x[i] - max_x);
    }
    return max_x + log(out);
}

void logistic_loss_grad(const double *w,
                        const double *x_tr,
                        const double *y_tr,
                        double *loss_grad,
                        double eta,
                        int n_samples,
                        int n_features) {
    for (int i = 0; i < n_features + 1; i++) {
        if (isnan(w[i])) {
            printf("%f\n", w[i]);
            printf("warning: loss grad error!\n");
            break;
        }
    }
    int i, n = n_samples, p = n_features;
    double intercept = w[p], sum_z0 = 0.0;
    loss_grad[0] = 0.0;
    double *yz = malloc(sizeof(double) * n);
    double *z0 = malloc(sizeof(double) * n);
    double *logistic = malloc(sizeof(double) * n);
    for (i = 0; i < n; i++) { /** calculate yz */
        yz[i] = intercept;
    }
    //x_tr^T*w+
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                n, p, 1., x_tr, p, w, 1, 1., yz, 1);
    for (i = 0; i < n; i++) {
        yz[i] *= y_tr[i];
    }
    expit(yz, z0, n); // calculate z0 and final intercept
    /** calculate logistic logistic[i] = 1/(1+exp(-y[i]*(xi^T*w+c)))*/
    log_logistic(yz, logistic, n);
    /** calculate loss of data fitting part*/
    for (i = 0; i < n; i++) {
        z0[i] = (z0[i] - 1.) * y_tr[i];
        sum_z0 += z0[i];
        loss_grad[0] -= logistic[i];
    }
    /**calculate loss of regularization part (it does not have intercept)*/
    loss_grad[0] += 0.5 * eta * cblas_ddot(p, w, 1, w, 1);
    /** calculate gradient of coefficients*/
    memcpy(loss_grad + 1, w, sizeof(double) * p);
    /** x^T*z0 + eta*w, where z0[i]=(logistic[i] - 1.)*yi*/
    cblas_dgemv(CblasRowMajor, CblasTrans,
                n, p, 1., x_tr, p, z0, 1, eta, loss_grad + 1, 1);
    /** calculate gradient of intercept part*/
    loss_grad[p + 1] = sum_z0; // intercept part
    free(logistic);
    free(z0);
    free(yz);
}

void logistic_loss_grad_sparse(const double *w,
                               const double *x_tr,
                               const double *y_tr,
                               double *loss_grad,
                               double eta,
                               int n_samples,
                               int n_features) {
    //sub_p is number of nonzeros in w.
    int i, n = n_samples, p = n_features, sub_p = 0;
    double intercept = w[p], sum_z0 = 0.0;
    loss_grad[0] = 0.0;
    double *yz = malloc(sizeof(double) * n);
    double *z0 = malloc(sizeof(double) * n);
    double *logistic = malloc(sizeof(double) * n);
    for (i = 0; i < n; i++) { /** calculate yz */
        yz[i] = intercept;
    }
    double nonzero_w[p];
    int nonzero_w_ind[p];
    for (i = 0; i < p; i++) {
        if (w[i] != 0.0) {
            nonzero_w_ind[sub_p] = i;
            nonzero_w[sub_p] = w[i];
            sub_p++;
        }
    }
    //x_tr^T*w+ to use the s of w
    for (i = 0; i < n; i++) {
        double tmp_val = 0.0;
        for (int j = 0; j < sub_p; j++) {
            tmp_val += x_tr[i * p + nonzero_w_ind[j]] * nonzero_w[j];
        }
        yz[i] = yz[i] + tmp_val;
    }
    for (i = 0; i < n; i++) {
        yz[i] *= y_tr[i];
    }
    expit(yz, z0, n); // calculate z0 and final intercept
    /** calculate logistic logistic[i] = 1/(1+exp(-y[i]*(xi^T*w+c)))*/
    log_logistic(yz, logistic, n);
    /** calculate loss of data fitting part*/
    for (i = 0; i < n; i++) {
        z0[i] = (z0[i] - 1.) * y_tr[i];
        sum_z0 += z0[i];
        loss_grad[0] -= logistic[i];
    }
    /**calculate loss of regularization part (it does not have intercept)*/
    loss_grad[0] += 0.5 * eta * cblas_ddot(p, w, 1, w, 1);
    /** calculate gradient of coefficients*/
    memcpy(loss_grad + 1, w, sizeof(double) * p);
    /** x^T*z0 + eta*w, where z0[i]=(logistic[i] - 1.)*yi*/
    cblas_dgemv(CblasRowMajor, CblasTrans,
                n, p, 1., x_tr, p, z0, 1, eta, loss_grad + 1, 1);
    /** calculate gradient of intercept part*/
    loss_grad[p + 1] = sum_z0; // intercept part
    free(logistic);
    free(z0);
    free(yz);
}


void logistic_predict(const double *x_te,
                      const double *wt,
                      double *pred_prob,
                      double *pred_label,
                      double threshold,
                      int n,
                      int p) {
    openblas_set_num_threads(1);
    int i;
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                n, p, 1., x_te, p, wt, 1, 0., pred_prob, 1);
    for (i = 0; i < n; i++) {
        pred_prob[i] += wt[p]; // intercept
    }
    expit(pred_prob, pred_prob, n);
    for (i = 0; i < n; i++) {
        if (pred_prob[i] >= threshold) {
            pred_label[i] = 1.;
        } else {
            pred_label[i] = -1.;
        }
    }
}

void ls_loss_grad(const double *w,
                  const double *x_tr,
                  const double *y_tr,
                  double *loss_grad,
                  double eta,
                  int n_samples,
                  int n_features) {
    for (int i = 0; i < n_features + 1; i++) {
        if (isnan(w[i])) {
            printf("%f\n", w[i]);
            printf("warning: loss grad error!\n");
            break;
        }
    }
    int i, n = n_samples, p = n_features;
    double *yz = malloc(sizeof(double) * n);
    double *w0 = malloc(sizeof(double) * n);
    cblas_dcopy(n, y_tr, 1, yz, 1); // y_tr --> yz
    cblas_dscal(p + 2, 0.0, loss_grad, 1);
    cblas_daxpy(p, eta, w, 1, loss_grad + 1, 1); // lambda_l2*w --> loss_grad
    //Order,TransA, M, N, alpha, A, lda, x, incX, beta, Y, incY
    // Y<-alpha*AX + beta*Y, where A=MxN, Y=N
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                n, p, 1., x_tr, p, w, 1, -1., yz, 1); //Xw - y
    for (i = 0; i < n; i++) { w0[i] = w[p]; }
    cblas_daxpy(n, 1., w0, 1, yz, 1);
    loss_grad[0] = cblas_ddot(n, yz, 1, yz, 1) / (2. * n);
    loss_grad[0] += (.5 * eta) * cblas_ddot(p, w, 1, w, 1); //loss
    cblas_dgemv(CblasRowMajor, CblasTrans,
                n, p, 1. / n, x_tr, p, yz, 1, 1., loss_grad + 1, 1);
    for (i = 0; i < n; i++) {
        w0[i] = w[p];
        loss_grad[p + 1] = yz[i];
    }
    loss_grad[p + 1] /= n;
    free(w0), free(yz);
}

void least_square_predict(const double *x_te,
                          const double *wt,
                          double *pred_prob,
                          double *pred_label,
                          double threshold,
                          int n_samples,
                          int p_features) {
    openblas_set_num_threads(1);
    int i, n = n_samples, p = p_features;
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                n, p, 1., x_te, p, wt, 1, 0., pred_prob, 1);
    for (i = 0; i < n; i++) {
        pred_prob[i] += wt[p]; // intercept
    }
    // this is not useful.
    for (i = 0; i < n; i++) {
        if (pred_prob[i] >= threshold) {
            pred_label[i] = 1.;
        } else {
            pred_label[i] = -1.;
        }
    }
}
