//
// Created by baojian on 8/11/18.
//

#ifndef FAST_PCST_SPARSE_ALGORITHMS_H
#define FAST_PCST_SPARSE_ALGORITHMS_H

#include "head_tail_proj.h"


typedef struct {
    double *x_tr;
    double *y_tr;
    double *w0;
    int p;
    int num_tr;
    int sparsity;
    int verbose;
    double gamma;
    double l2_lambda;
    int loss_func;
    int *best_subset;
    int best_subset_len;
} best_subset;


typedef struct {
    double *x_tr;
    double *y_tr;
    EdgePair *edges;
    double *weights;
    double *w0;
    int m;
    int p;
    int num_tr;
    int sparsity_low;
    int sparsity_high;
    int max_num_iter;
    int g;
    int root;
    int verbose;
    double lr;
    double l2_lambda;
    int loss_func;
} graph_iht_para;

typedef struct {
    EdgePair *edges;
    double *costs;
    double *prizes;
    int m;
    int p;
    int num_tr;
    int sparsity_low;
    int sparsity_high;
    int max_num_iter;
    int g;
    int root;
    int verbose;
} head_tail_binsearch_para;


typedef struct {
    double *wt;
    double *wt_bar;
    double *losses;
    double total_time;
    Array *re_nodes;
    Array *re_edges;
    int num_pcst;
    double run_time_head;
    double run_time_tail;
} ReStat;

typedef struct {
    double *wt;
    double *wt_bar;
    int *nonzeros_wt;
    int *nonzeros_wt_bar;
    double *losses;
    double total_time;
    Array *re_nodes;
    Array *re_edges;
    int num_pcst;
    double run_time_head;
    double run_time_tail;
} StochasticStat;


bool algo_batch_iht_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s,
        int p, int num_tr, double lr, double eta, int max_iter, double tol,
        int verbose, double *wt, double *losses, double *total_time);

bool algo_batch_graph_iht_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, double *losses, double *run_time_head,
        double *run_time_tail, double *total_time);


#endif //FAST_PCST_SPARSE_ALGORITHMS_H
