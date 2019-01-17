//
// Created by baojian on 11/22/18.
//

#include "algo_conjugate_grad.h"

bool cg_solve(double *x_tr, double *b, double tol, int max_iter, int verbose,
              int n, double *x, double *res, int *iter) {
    double *q = malloc(sizeof(double) * n);
    for (int ii = 0; ii < n; ii++) {
        x[ii] = 0.0;
        q[ii] = 0.0;
    }
    double *r = b;
    double *d = r;
    double delta = cblas_ddot(n, r, 1, r, 1);
    double delta0 = cblas_ddot(n, b, 1, b, 1);
    int num_iter = 0;
    double *best_x = malloc(sizeof(double) * n);
    cblas_dcopy(n, best_x, 1, x, 1);
    double best_res = sqrt(delta / delta0);

    while ((num_iter < max_iter) && (delta > (tol * tol * delta0))) {
        // q = A*d
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    n, n, 1., x_tr, n, d, 1, 1., q, 1);
        double alpha = delta / (cblas_ddot(n, d, 1, q, 1));
        cblas_daxpy(n, alpha, d, 1, x, 1); // alpha*x + y --> y
        if (fmod(num_iter, 50) == 0.0) {
            // r = b - Aux*x
            cblas_dcopy(n, r, 1, b, 1);
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        n, n, -1., x_tr, n, x, 1, 1., r, 1);
        } else {
            cblas_daxpy(n, -alpha, x, 1, r, 1);
        }
        double delta_old = delta;
        delta = cblas_ddot(n, r, 1, r, 1);
        double beta = delta / delta_old;
        cblas_ccopy(n, r, 1, d, 1);
        cblas_daxpy(n, beta, d, 1, r, 1);
        if (sqrt(delta / delta0) < best_res) {
            cblas_dcopy(n, best_x, 1, x, 1);
            best_res = sqrt(delta / delta0);
        }
        if ((verbose) & (fmod(num_iter, verbose) == 0))
            printf("cg: Iter = %d, "
                   "Best residual = %8.3e, "
                   "Current residual = %8.3e\n",
                   num_iter, best_res, sqrt(delta / delta0));
    }
    if (verbose >= 1) {
        printf("cg: Iterations = %d, best residual = %14.8e\n",
               num_iter, best_res);
    }
    cblas_dcopy(n, x, 1, best_x, 1);
    *res = best_res;
    *iter = num_iter;
    free(best_x);
    free(q);
    return true;
}