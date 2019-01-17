//
// Created by baojian on 8/11/18.
//
#include <cblas.h>
#include "sort.h"
#include "loss.h"
#include "sparse_algorithms.h"

#define sign(x) ((x > 0) -(x < 0))


StochasticStat *make_stochastic_stat(int p, int num_tr) {
    StochasticStat *stat = malloc(sizeof(StochasticStat));
    stat->wt = malloc(sizeof(double) * (p + 1));
    stat->wt_bar = malloc(sizeof(double) * (p + 1));
    stat->nonzeros_wt = malloc(sizeof(int) * num_tr);
    stat->nonzeros_wt_bar = malloc(sizeof(int) * num_tr);
    stat->losses = malloc(sizeof(double) * num_tr);
    for (int i = 0; i < num_tr; i++) {
        stat->nonzeros_wt[i] = 0;
        stat->nonzeros_wt_bar[i] = 0;
    }
    stat->total_time = 0.0;
    stat->num_pcst = 0;
    stat->run_time_head = 0.0;
    stat->run_time_tail = 0.0;
    stat->re_nodes = malloc(sizeof(Array));
    stat->re_nodes->size = 0;
    stat->re_nodes->array = malloc(sizeof(int) * p);
    stat->re_edges = malloc(sizeof(Array));
    stat->re_edges->size = 0;
    stat->re_edges->array = malloc(sizeof(int) * p);
    return stat;
}

bool free_stochastic_stat(StochasticStat *online_stat) {
    free(online_stat->losses);
    free(online_stat->nonzeros_wt);
    free(online_stat->nonzeros_wt_bar);
    free(online_stat->wt);
    free(online_stat->wt_bar);
    free(online_stat->re_nodes->array);
    free(online_stat->re_nodes);
    free(online_stat->re_edges->array);
    free(online_stat->re_edges);
    free(online_stat);
    return true;
}

void min_f_posi(const Array *proj_nodes, const double *x_tr,
                const double *y_tr, int max_iter, double eta, double *wt,
                int n, int p) {
    openblas_set_num_threads(1);
    int i;
    double *loss_grad = (double *) malloc((p + 2) * sizeof(double));
    double *tmp_loss_grad = (double *) malloc((p + 2) * sizeof(double));
    double *wt_tmp = (double *) malloc((p + 1) * sizeof(double));
    /**
     * make sure the start point is a feasible point. here we do a trick:
     * we treat wt as an initial point itself. and of course wt is always a
     * feasible point. A Frank-Wolfe style minimization with
     * backtracking line search. Other algorithms can be considered:
     * Newton's method, trust region, etc.
     */
    double beta, lr, grad_sq;
    for (i = 0; i < max_iter; i++) {
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, n, p);
        beta = 0.8, lr = 1.0;
        grad_sq = 0.5 * cblas_ddot(p + 1, loss_grad + 1, 1, loss_grad + 1, 1);
        while (true) { // using backtracking line search
            cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
            cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
            logistic_loss_grad(wt_tmp, x_tr, y_tr, tmp_loss_grad, eta, n, p);
            if (tmp_loss_grad[0] > (loss_grad[0] - lr * grad_sq)) {
                lr *= beta;
            } else {
                break;
            }
        }
        // projection step
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int k = 0; k < proj_nodes->size; k++) {
            wt_tmp[proj_nodes->array[k]] = wt[proj_nodes->array[k]];
            // positive constraint.
            if (wt_tmp[proj_nodes->array[k]] < 0.) {
                wt_tmp[proj_nodes->array[k]] = 0.0;
            }
        }
        wt_tmp[p] = wt[p];
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
    }
    free(wt_tmp);
    free(tmp_loss_grad);
    free(loss_grad);
}

void min_f(const Array *proj_nodes, const double *x_tr,
           const double *y_tr, int max_iter, double eta, double *wt,
           int n, int p) {
    openblas_set_num_threads(1);
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *tmp_loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    double beta, lr, grad_sq;
    /**
     * make sure the start point is a feasible point. here we do a trick:
     * we treat wt as an initial point itself. and of course wt is always a
     * feasible point. A Frank-Wolfe style minimization with
     * backtracking line search. Other algorithms can be considered:
     * Newton's method, trust region, etc.
     */
    for (int i = 0; i < max_iter; i++) {
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, n, p);
        beta = 0.8, lr = 1.0;
        grad_sq = 0.5 * cblas_ddot(p + 1, loss_grad + 1, 1, loss_grad + 1, 1);
        while (true) { // using backtracking line search
            cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
            cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
            logistic_loss_grad(wt_tmp, x_tr, y_tr, tmp_loss_grad, eta, n, p);
            if (tmp_loss_grad[0] > (loss_grad[0] - lr * grad_sq)) {
                lr *= beta;
            } else {
                break;
            }
        }
        // projection step
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int k = 0; k < proj_nodes->size; k++) {
            wt_tmp[proj_nodes->array[k]] = wt[proj_nodes->array[k]];
        }
        wt_tmp[p] = wt[p];
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
    }
    free(wt_tmp);
    free(tmp_loss_grad);
    free(loss_grad);
}

void min_f_sparse(
        const Array *proj_nodes, const double *x_tr,
        const double *y_tr, int max_iter, double eta, double *wt, int n,
        int p) {
    openblas_set_num_threads(1);
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *tmp_loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    double beta, lr, grad_sq;
    /**
     * make sure the start point is a feasible point. here we do a trick:
     * we treat wt as an initial point itself. and of course wt is always a
     * feasible point. A Frank-Wolfe style minimization with
     * backtracking line search. Other algorithms can be considered:
     * Newton's method, trust region, etc.
     */
    for (int i = 0; i < max_iter; i++) {
        logistic_loss_grad_sparse(
                wt, x_tr, y_tr, loss_grad, eta, n, p);
        beta = 0.8, lr = 1.0;
        grad_sq = 0.5 * cblas_ddot(p + 1, loss_grad + 1, 1, loss_grad + 1, 1);
        while (true) { // using backtracking line search
            cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
            cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
            logistic_loss_grad_sparse(
                    wt_tmp, x_tr, y_tr, tmp_loss_grad, eta, n, p);
            if (tmp_loss_grad[0] > (loss_grad[0] - lr * grad_sq)) {
                lr *= beta;
            } else {
                break;
            }
        }
        // projection step
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int k = 0; k < proj_nodes->size; k++) {
            wt_tmp[proj_nodes->array[k]] = wt[proj_nodes->array[k]];
        }
        wt_tmp[p] = wt[p];
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
    }
    free(wt_tmp);
    free(tmp_loss_grad);
    free(loss_grad);
}


bool algo_batch_iht_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s,
        int p, int num_tr, double lr, double eta, int max_iter, double tol,
        int verbose, double *wt, double *losses, double *total_time) {
    openblas_set_num_threads(1);
    int i;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> x_hat
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    int *sorted_ind = malloc(sizeof(double) * p);
    Array *re_nodes = malloc(sizeof(Array));
    re_nodes->array = malloc(sizeof(int) * p);
    clock_t start_total = clock();
    if (verbose > 0) {
        printf("learning rate: %lf, n: %d ,p: %d\n", lr, num_tr, p);
    }
    for (int tt = 0; tt < max_iter; tt++) {
        // each time has one sample.
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, num_tr, p);
        losses[tt] = loss_grad[0];
        if (verbose > 0) {
            printf("losses[%d]:%.6f s:%d ", tt, losses[tt], s);
        }
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // x_hat --> wt_tmp
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
        arg_magnitude_sort_descend(wt_tmp, sorted_ind, p);
        for (i = 0; i < s; i++) {
            wt[sorted_ind[i]] = wt_tmp[sorted_ind[i]];
        }
        wt[p] = wt_tmp[p];
        if (tt >= 1 && (fabs(losses[tt] - losses[tt - 1]) < tol)) {
            break; // stop earlier when it almost stops decreasing the loss
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt_tmp), free(sorted_ind);
    free(re_nodes->array), free(re_nodes);
    return true;
}


bool algo_batch_graph_iht_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, double *losses, double *run_time_head,
        double *run_time_tail, double *total_time) {
    openblas_set_num_threads(1);
    GraphStat *head_stat = make_graph_stat(p, m);
    GraphStat *tail_stat = make_graph_stat(p, m);
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> x_hat
    double *loss_grad = malloc(sizeof(double) * (p + 2));
    double *wt_tmp = malloc(sizeof(double) * (p + 1));
    double *tmp_prizes = malloc(sizeof(double) * p);
    double *tmp_costs = malloc(sizeof(double) * m);
    clock_t start_head, start_tail, start_total = clock();
    double C = 2. * (s - 1.), delta = 1. / 169., nu = 2.5, err_tol = 1e-6;
    double budget = (s - 1.), eps = 1e-6;
    int i, root = -1;
    enum PruningMethod pruning = GWPruning;
    for (i = 0; i < m; i++) { tmp_costs[i] = costs[i] + budget / (double) s; }
    for (int tt = 0; tt < max_iter; tt++) {
        // each time has one sample.
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, num_tr, p);
        losses[tt] = loss_grad[0];
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = loss_grad[i + 1] * loss_grad[i + 1];
        }
        start_head = clock();
        head_proj_exact(edges, tmp_costs, tmp_prizes, g, C, delta, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, head_stat);
        cblas_dcopy(p + 1, loss_grad + 1, 1, wt_tmp, 1); // x_hat --> wt_tmp
        cblas_dscal(p + 1, 0.0, loss_grad + 1, 1);
        for (i = 0; i < re_nodes->size; i++) {
            int cur_node = re_nodes->array[i];
            loss_grad[cur_node + 1] = wt_tmp[cur_node];
        }
        *run_time_head += ((double) (clock() - start_head)) / CLOCKS_PER_SEC;
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // x_hat --> wt_tmp
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = wt_tmp[i] * wt_tmp[i];
        }
        start_tail = clock();
        tail_proj_exact(edges, tmp_costs, tmp_prizes, g, C, nu, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, tail_stat);
        *run_time_tail += ((double) (clock() - start_tail)) / CLOCKS_PER_SEC;
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (i = 0; i < re_nodes->size; i++) {
            int cur_node = re_nodes->array[i];
            wt[cur_node] = wt_tmp[cur_node];
        }
        wt[p] = wt_tmp[p];
        if ((tt >= 1 && (fabs(losses[tt] - losses[tt - 1]) < tol)) ||
            (tt >= 1 && (losses[tt] >= losses[tt - 1]))) {
            break; // stop earlier when it almost stops decreasing the loss
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(tmp_costs), free(tmp_prizes), free(loss_grad), free(wt_tmp);
    return true;
}