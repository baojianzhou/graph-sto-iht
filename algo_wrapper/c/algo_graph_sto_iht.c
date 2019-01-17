//
// Created by baojian on 10/15/18.
//
#include <cblas.h>
#include "loss.h"
#include "sort.h"
#include "algo_graph_sto_iht.h"


bool algo_online_graph_iht_logit(graph_iht_para *para, ReStat *stat) {

    clock_t start_time = clock();
    openblas_set_num_threads(1);

    int p = para->p;
    int m = para->m;
    int verbose = para->verbose;
    double *x_tr = para->x_tr;
    double *y_tr = para->y_tr;
    double lr = para->lr;
    double *costs = para->weights;
    int root = para->root;
    int sparsity_low = para->sparsity_low;
    int sparsity_high = para->sparsity_high;
    int max_num_iter = para->max_num_iter;
    EdgePair *edges = para->edges;
    int g = para->g;
    double l2_lambda = para->l2_lambda;
    double *wt, *wt_bar, *loss_grad, *wt_tmp, *proj_prizes, *proj_costs;

    GraphStat *graph_stat = make_graph_stat(p, m);

    wt = malloc(sizeof(double) * (p + 1));
    wt_bar = malloc(sizeof(double) * (p + 1));
    loss_grad = malloc(sizeof(double) * (p + 2));
    wt_tmp = malloc(sizeof(double) * (p + 1));
    proj_prizes = malloc(sizeof(double) * p);
    proj_costs = malloc(sizeof(double) * m);
    cblas_dcopy(para->p + 1, para->w0, 1, wt, 1);      // w0 --> x_hat
    cblas_dcopy(para->p + 1, para->w0, 1, wt_bar, 1);  // w0 --> x_bar

    for (int tt = 0; tt < para->num_tr; tt++) {
        // 1.   unlabeled example x_i = x_tr + tt*p arrives and
        //      make prediction based on existing costs x_hat, x_bar
        double *x_i = x_tr + tt * p, *y_i = y_tr + tt;
        // 2. observe y_i = y_tr + tt and have some loss
        logistic_loss_grad(wt, x_i, y_i, loss_grad, l2_lambda, 1, p);
        stat->losses[tt] = loss_grad[0];
        // 3. graph projection: update x_hat= x_hat - lr/sqrt(tt) * head(f_xi, x_hat)
        for (int i = 0; i < p; i++) {
            proj_prizes[i] = loss_grad[i + 1] * loss_grad[i + 1];
        }
        head_tail_binsearch(
                edges, costs, proj_prizes, p, m, g, root, sparsity_low,
                sparsity_high, max_num_iter, GWPruning, verbose,
                graph_stat);
        cblas_dcopy(p + 1, loss_grad + 1, 1, wt_tmp, 1); // x_hat --> wt_tmp
        cblas_dscal(p + 1, 0.0, loss_grad + 1, 1);
        for (int i = 0; i < graph_stat->re_nodes->size; i++) {
            int cur_node = graph_stat->re_nodes->array[i];
            loss_grad[cur_node + 1] = wt_tmp[cur_node];
        }
        stat->run_time_head += graph_stat->run_time;
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // x_hat --> wt_tmp
        cblas_daxpy(p + 1, -lr / sqrt(tt + 1.), loss_grad + 1, 1, wt_tmp, 1);
        for (int i = 0; i < p; i++) {
            proj_prizes[i] = wt_tmp[i] * wt_tmp[i];
        }
        head_tail_binsearch(
                edges, costs, proj_prizes, p, m, g, root, sparsity_low,
                sparsity_high, max_num_iter, GWPruning, verbose,
                graph_stat);
        stat->run_time_tail += graph_stat->run_time;
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (int i = 0; i < graph_stat->re_nodes->size; i++) {
            int cur_node = graph_stat->re_nodes->array[i];
            wt[cur_node] = wt_tmp[cur_node];
        }
        wt[p] = wt_tmp[p];
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
    }
    stat->re_nodes->size = graph_stat->re_nodes->size;
    stat->re_edges->size = graph_stat->re_edges->size;
    for (int i = 0; i < graph_stat->re_nodes->size; i++) {
        stat->re_nodes->array[i] = graph_stat->re_nodes->array[i];
    }
    for (int i = 0; i < graph_stat->re_edges->size; i++) {
        stat->re_edges->array[i] = graph_stat->re_edges->array[i];
    }
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(wt);
    free(wt_bar);
    free(wt_tmp);
    free(proj_costs);
    free(proj_prizes);
    free(loss_grad);
    free_graph_stat(graph_stat);
    return true;
}


bool algo_online_graph_iht_least_square(
        graph_iht_para *para, ReStat *stat) {

    clock_t start_time = clock();
    openblas_set_num_threads(1);

    int p = para->p;
    int m = para->m;
    int verbose = para->verbose;
    double *x_tr = para->x_tr;
    double *y_tr = para->y_tr;
    double lr = para->lr;
    double *costs = para->weights;
    int root = para->root;
    int sparsity_low = para->sparsity_low;
    int sparsity_high = para->sparsity_high;
    int max_num_iter = para->max_num_iter;
    EdgePair *edges = para->edges;
    int g = para->g;
    double l2_lambda = para->l2_lambda;
    double *wt, *wt_bar, *loss_grad, *wt_tmp, *proj_prizes, *proj_costs;

    GraphStat *graph_stat = make_graph_stat(p, m);

    wt = malloc(sizeof(double) * (p + 1));
    wt_bar = malloc(sizeof(double) * (p + 1));
    loss_grad = malloc(sizeof(double) * (p + 2));
    wt_tmp = malloc(sizeof(double) * (p + 1));
    proj_prizes = malloc(sizeof(double) * p);
    proj_costs = malloc(sizeof(double) * m);
    cblas_dcopy(para->p + 1, para->w0, 1, wt, 1);      // w0 --> x_hat
    cblas_dcopy(para->p + 1, para->w0, 1, wt_bar, 1);  // w0 --> x_bar

    for (int tt = 0; tt < para->num_tr; tt++) {
        // 1.   unlabeled example x_i = x_tr + tt*p arrives and
        //      make prediction based on existing costs x_hat, x_bar
        double *x_i = x_tr + tt * p, *y_i = y_tr + tt;
        // 2. observe y_i = y_tr + tt and have some loss
        ls_loss_grad(wt, x_i, y_i, loss_grad, l2_lambda, 1, p);
        stat->losses[tt] = loss_grad[0];
        // 3. graph projection: update x_hat= x_hat - lr/sqrt(tt) * head(f_xi, x_hat)
        for (int i = 0; i < p; i++) {
            proj_prizes[i] = loss_grad[i + 1] * loss_grad[i + 1];
        }
        head_tail_binsearch(
                edges, costs, proj_prizes, p, m, g, root, sparsity_low,
                sparsity_high, max_num_iter, GWPruning, verbose,
                graph_stat);
        cblas_dcopy(p + 1, loss_grad + 1, 1, wt_tmp, 1); // x_hat --> wt_tmp
        cblas_dscal(p + 1, 0.0, loss_grad + 1, 1);
        for (int i = 0; i < graph_stat->re_nodes->size; i++) {
            int cur_node = graph_stat->re_nodes->array[i];
            loss_grad[cur_node + 1] = wt_tmp[cur_node];
        }
        stat->run_time_head += graph_stat->run_time;
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // x_hat --> wt_tmp
        cblas_daxpy(p + 1, -lr / sqrt(tt + 1.), loss_grad + 1, 1, wt_tmp, 1);
        for (int i = 0; i < p; i++) {
            proj_prizes[i] = wt_tmp[i] * wt_tmp[i];
        }
        head_tail_binsearch(
                edges, costs, proj_prizes, p, m, g, root, sparsity_low,
                sparsity_high, max_num_iter, GWPruning, verbose,
                graph_stat);
        stat->run_time_tail += graph_stat->run_time;
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (int i = 0; i < graph_stat->re_nodes->size; i++) {
            int cur_node = graph_stat->re_nodes->array[i];
            wt[cur_node] = wt_tmp[cur_node];
        }
        wt[p] = wt_tmp[p];
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
    }
    stat->re_nodes->size = graph_stat->re_nodes->size;
    stat->re_edges->size = graph_stat->re_edges->size;
    for (int i = 0; i < graph_stat->re_nodes->size; i++) {
        stat->re_nodes->array[i] = graph_stat->re_nodes->array[i];
    }
    for (int i = 0; i < graph_stat->re_edges->size; i++) {
        stat->re_edges->array[i] = graph_stat->re_edges->array[i];
    }
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(wt);
    free(wt_bar);
    free(wt_tmp);
    free(proj_costs);
    free(proj_prizes);
    free(loss_grad);
    free_graph_stat(graph_stat);
    return true;
}
