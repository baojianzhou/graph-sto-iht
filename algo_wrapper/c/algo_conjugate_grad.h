//
// Created by baojian on 11/22/18.
//

#ifndef ONLINE_OPT_ALGO_CONJUGATE_GRAD_H
#define ONLINE_OPT_ALGO_CONJUGATE_GRAD_H

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <cblas.h>

/**
 * solve Ax = b via conjugate gradient, where A is nxn and b is nx1
 * This is an c version of cgsolve which is originally written
 *  by: Justin Romberg, Caltech using Matlab.
 * @param x_tr NxN Either an NxN matrix, or a function handle.
 * @param b N vector
 * @param tol Algorithm terminates when norm(Ax-b)/norm(b) < tol .(e.g. 1e-4)
 * @param max_iter Maximum number of iterations. (max_iter=100)
 * @param verbose If 0, do not print out progress messages. If and integer
 *          greater than 0, print out progress every 'verbose' iters.
 * @param x
 * @param res
 * @param iter
 * @return
 */
bool cg_solve(double *x_tr, double *b, double tol, int max_iter, int verbose,
              int n, double *x, double *res, int *iter);

#endif //ONLINE_OPT_ALGO_CONJUGATE_GRAD_H
