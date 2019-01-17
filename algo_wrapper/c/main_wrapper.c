//
// Created by baojian on 8/11/18.
//
#include <Python.h>
#include <numpy/arrayobject.h>
#include "algo_sto_iht.h"
#include "algo_best_subset.h"
#include "algo_graph_sto_iht.h"

bool get_data(
        int n, int p, int m, double *x_tr, double *y_tr, double *w0,
        EdgePair *edges, double *weights, PyArrayObject *x_tr_,
        PyArrayObject *y_tr_, PyArrayObject *w0_, PyArrayObject *edges_,
        PyArrayObject *weights_) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            x_tr[i * p + j] = *(double *) PyArray_GETPTR2(x_tr_, i, j);
        }
        y_tr[i] = *(double *) PyArray_GETPTR1(y_tr_, i);
    }
    for (i = 0; i < (p + 1); i++) {
        w0[i] = *(double *) PyArray_GETPTR1(w0_, i);;
    }
    if (edges != NULL) {
        for (i = 0; i < m; i++) {
            edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
            edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
            weights[i] = *(double *) PyArray_GETPTR1(weights_, i);
        }
    }
    return true;
}

PyObject *batch_get_result(
        int p, int max_iter, double total_time, double *wt, double *losses) {
    PyObject *results = PyTuple_New(3);
    PyObject *re_wt = PyList_New(p + 1);
    for (int i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
    }
    PyObject *re_losses = PyList_New(max_iter);
    for (int i = 0; i < max_iter; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_losses);
    PyTuple_SetItem(results, 2, re_total_time);
    return results;
}

PyObject *online_get_result(int p, int num_tr, ReStat *stat) {
    PyObject *results = PyTuple_New(11);

    PyObject *re_wt = PyList_New(p + 1);
    PyObject *re_wt_bar = PyList_New(p + 1);
    PyObject *re_nodes = PyList_New(stat->re_nodes->size);
    PyObject *re_edges = PyList_New(stat->re_edges->size);
    PyObject *re_num_pcst = PyInt_FromLong(stat->num_pcst);
    PyObject *re_losses = PyList_New(num_tr);
    PyObject *re_run_time_head = PyFloat_FromDouble(stat->run_time_head);
    PyObject *re_run_time_tail = PyFloat_FromDouble(stat->run_time_tail);
    PyObject *re_missed_wt = PyList_New(num_tr);
    PyObject *re_missed_wt_bar = PyList_New(num_tr);
    PyObject *re_total_time = PyFloat_FromDouble(stat->total_time);
    for (int i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(stat->wt[i]));
        PyList_SetItem(re_wt_bar, i, PyFloat_FromDouble(stat->wt_bar[i]));
    }
    for (int i = 0; i < stat->re_nodes->size; i++) {
        PyList_SetItem(re_nodes, i, PyInt_FromLong(stat->re_nodes->array[i]));
    }
    for (int i = 0; i < stat->re_edges->size; i++) {
        PyList_SetItem(re_edges, i, PyInt_FromLong(stat->re_edges->array[i]));
    }
    for (int i = 0; i < num_tr; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(stat->losses[i]));
    }
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_wt_bar);
    PyTuple_SetItem(results, 2, re_losses);
    PyTuple_SetItem(results, 3, re_missed_wt);
    PyTuple_SetItem(results, 4, re_missed_wt_bar);
    PyTuple_SetItem(results, 5, re_total_time);
    PyTuple_SetItem(results, 6, re_nodes);
    PyTuple_SetItem(results, 7, re_edges);
    PyTuple_SetItem(results, 8, re_num_pcst);
    PyTuple_SetItem(results, 9, re_run_time_head);
    PyTuple_SetItem(results, 10, re_run_time_tail);
    return results;
}

static PyObject *wrap_head_tail_binsearch(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    head_tail_binsearch_para *para = malloc(sizeof(head_tail_binsearch_para));
    PyArrayObject *edges_, *costs_, *prizes_;
    if (!PyArg_ParseTuple(args, "O!O!O!iiiiii",
                          &PyArray_Type, &edges_,
                          &PyArray_Type, &prizes_,
                          &PyArray_Type, &costs_,
                          &para->g,
                          &para->root,
                          &para->sparsity_low,
                          &para->sparsity_high,
                          &para->max_num_iter,
                          &para->verbose)) { return NULL; }

    para->p = (int) prizes_->dimensions[0];
    para->m = (int) edges_->dimensions[0];
    para->prizes = (double *) PyArray_DATA(prizes_);
    para->costs = (double *) PyArray_DATA(costs_);
    para->edges = malloc(sizeof(EdgePair) * para->m);
    for (int i = 0; i < para->m; i++) {
        para->edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
        para->edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
    }
    GraphStat *graph_stat = make_graph_stat(para->p, para->m);
    head_tail_binsearch(
            para->edges, para->costs, para->prizes, para->p, para->m, para->g,
            para->root, para->sparsity_low, para->sparsity_high,
            para->max_num_iter, GWPruning, para->verbose, graph_stat);
    PyObject *results = PyTuple_New(1);
    PyObject *re_nodes = PyList_New(graph_stat->re_nodes->size);
    for (int i = 0; i < graph_stat->re_nodes->size; i++) {
        int cur_node = graph_stat->re_nodes->array[i];
        PyList_SetItem(re_nodes, i, PyInt_FromLong(cur_node));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    free_graph_stat(graph_stat);
    free(para->edges);
    free(para);
    return results;
}

static PyObject *algo_sto_iht(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.^v^\n");
        return NULL;
    }
    StoIHTPara *para = malloc(sizeof(StoIHTPara));
    PyArrayObject *x_tr_, *y_tr_, *x0_, *x_star_, *prob_arr_;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiiiiiiidddd",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &x0_,
                          &PyArray_Type, &x_star_,
                          &PyArray_Type, &prob_arr_,
                          &para->n,
                          &para->p,
                          &para->s,
                          &para->b,
                          &para->loss_func,
                          &para->max_epochs,
                          &para->with_replace,
                          &para->verbose,
                          &para->lr,
                          &para->lambda_l2,
                          &para->tol_algo,
                          &para->tol_rec)) { return NULL; }

    bool flag01 = para->n == (int) x_tr_->dimensions[0];
    bool flag02 = para->p == (int) x_tr_->dimensions[1];
    bool flag03 = (para->b > 0 && para->s > 0);
    bool flag04 = (para->p > 0 && para->n > 0);
    bool flag05;
    if (para->loss_func == LeastSquare) {
        // without intercept
        flag05 = para->p == (int) x0_->dimensions[0];
    } else {
        flag05 = (para->p + 1) == (int) x0_->dimensions[0];
    }
    if (!(flag01 && flag02 && flag03 && flag04 && flag05)) {
        printf("too bad: parameters are inconsistent or have errors.");
        exit(0);
    }
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(x0_);
    para->x_star = (double *) PyArray_DATA(x_star_);
    para->prob_arr = (double *) PyArray_DATA(prob_arr_);

    ReStoIHTStat *stat = make_re_sto_iht_stat(
            para->p, para->b, para->max_epochs);
    if (para->loss_func == 1) {
        algo_sto_iht_logit(para, stat);
    } else if (para->loss_func == 0) {
        algo_sto_iht_least_square(para, stat);
    }

    PyObject *results = PyTuple_New(6);
    PyObject *re_x_hat = PyList_New(para->p);
    PyObject *re_x_bar = PyList_New(para->p);
    PyObject *re_epochs_losses = PyList_New(para->max_epochs);
    PyObject *re_x_epochs_errors = PyList_New(para->max_epochs);
    PyObject *re_run_time = PyFloat_FromDouble(stat->run_time);
    PyObject *re_num_epochs = PyLong_FromLong(stat->num_epochs);
    for (int i = 0; i < para->p; i++) {
        PyList_SetItem(re_x_hat, i,
                       PyFloat_FromDouble(stat->x_hat[i]));
        PyList_SetItem(re_x_bar, i,
                       PyFloat_FromDouble(stat->x_bar[i]));
    }
    for (int i = 0; i < para->max_epochs; i++) {
        PyList_SetItem(re_epochs_losses, i,
                       PyFloat_FromDouble(stat->epoch_losses[i]));
        PyList_SetItem(re_x_epochs_errors, i,
                       PyFloat_FromDouble(stat->x_epoch_errors[i]));
    }
    PyTuple_SetItem(results, 0, re_x_hat);
    PyTuple_SetItem(results, 1, re_x_bar);
    PyTuple_SetItem(results, 2, re_epochs_losses);
    PyTuple_SetItem(results, 3, re_x_epochs_errors);
    PyTuple_SetItem(results, 4, re_run_time);
    PyTuple_SetItem(results, 5, re_num_epochs);
    free_re_sto_iht_stat(stat);
    return results;
}


//input:x_tr, y_tr, w0, edges, costs, lr, lambda_l2, g,
//            sparsity_low, sparsity_high, root, max_num_iter, 0, verbose
static PyObject *algo_graph_sto_iht(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    graph_iht_para *para = malloc(sizeof(graph_iht_para));
    PyArrayObject *x_tr_, *y_tr_, *w0_, *edges_, *weights_;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!ddiiiiiii",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &w0_,
                          &PyArray_Type, &edges_,
                          &PyArray_Type, &weights_,
                          &para->lr,
                          &para->l2_lambda,
                          &para->g,
                          &para->sparsity_low,
                          &para->sparsity_high,
                          &para->root,
                          &para->max_num_iter,
                          &para->loss_func,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];      // # of samples
    para->p = (int) x_tr_->dimensions[1];   // # of features/nodes
    para->m = (int) edges_->dimensions[0];  // # of edges
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    para->weights = (double *) PyArray_DATA(weights_);
    para->edges = malloc(sizeof(EdgePair) * para->m);
    for (int i = 0; i < para->m; i++) {
        para->edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
        para->edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
    }
    ReStat *stat = make_re_sto_iht_stat(para->p, para->num_tr, 0);
    if (para->loss_func == 0) {
        algo_online_graph_iht_logit(para, stat);
    } else if (para->loss_func == 1) {
        algo_online_graph_iht_least_square(para, stat);
    }
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para->edges), free(para), free_re_sto_iht_stat(stat);
    return results;
}


// x_tr, y_tr, w0,best_subset, gamma, l2_lambda_, s, 0, verbose
static PyObject *algo_best_subset(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    best_subset *para = malloc(sizeof(best_subset));
    PyArrayObject *x_tr_, *y_tr_, *w0_, *best_subset_;
    if (!PyArg_ParseTuple(args, "O!O!O!O!ddiii",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &w0_,
                          &PyArray_Type, &best_subset_,
                          &para->gamma,
                          &para->l2_lambda,
                          &para->sparsity,
                          &para->loss_func,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];      // # of samples
    para->p = (int) x_tr_->dimensions[1];   // # of features/nodes
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    para->best_subset_len = (int) best_subset_->dimensions[0];
    para->best_subset = (int *) PyArray_DATA(best_subset_);
    ReStat *stat = make_re_sto_iht_stat(para->p, para->num_tr, 0);
    if (para->loss_func == 0) {
        algo_online_best_subset_logit(para, stat);
    } else if (para->loss_func == 1) {
        algo_online_best_subset_logit(para, stat);
    } else {
        algo_online_best_subset_logit(para, stat);
    }
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para), free_re_sto_iht_stat(stat);
    return results;
}


static PyObject *batch_iht_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, sparsity, verbose, max_iter;
    double lr, eta, tol;
    if (!PyArg_ParseTuple(
            args, "O!O!O!dididi", &PyArray_Type, &x_tr_, &PyArray_Type, &y_tr_,
            &PyArray_Type, &w0_, &lr, &sparsity, &tol, &max_iter, &eta,
            &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = malloc(sizeof(double) * (n * p));
    double *y_tr = malloc(sizeof(double) * n);
    double *w0 = malloc(sizeof(double) * (p + 1));
    get_data(n, p, -1, x_tr, y_tr, w0, NULL, NULL,
             x_tr_, y_tr_, w0_, NULL, NULL);
    double *wt = malloc(sizeof(double) * (p + 1));
    double *losses = malloc(sizeof(double) * max_iter);
    double total_time = 0.0;
    if (sparsity > p) {
        printf("the parameter of s is too large!\n");
        printf("s is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    algo_batch_iht_logit(x_tr, y_tr, w0, sparsity, p, n, lr, eta, max_iter,
                         tol, verbose, wt, losses, &total_time);
    PyObject *results = batch_get_result(p, max_iter, total_time, wt, losses);
    free(losses), free(x_tr), free(y_tr), free(w0), free(wt);
    return results;
}


static PyObject *batch_graph_iht_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    EdgePair *edges;
    int g = 1, n, p, m, sparsity, verbose, max_iter;
    double lr, eta, tol, *weights, *x_tr, *y_tr, *w0, *wt, *losses;
    PyArrayObject *x_tr_, *y_tr_, *w0_, *edges_, *weights_;
    if (!PyArg_ParseTuple(
            args, "O!O!O!dididO!O!i", &PyArray_Type, &x_tr_, &PyArray_Type,
            &y_tr_, &PyArray_Type, &w0_, &lr, &sparsity, &tol, &max_iter, &eta,
            &PyArray_Type, &edges_, &PyArray_Type, &weights_,
            &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);       // number of samples
    p = (int) (x_tr_->dimensions[1]);       // number of features
    m = (int) edges_->dimensions[0];        // number of edges
    x_tr = malloc(sizeof(double) * (n * p));
    y_tr = malloc(sizeof(double) * n);
    w0 = malloc(sizeof(double) * (p + 1));
    edges = malloc(sizeof(EdgePair) * m);
    weights = malloc(sizeof(double) * m);
    get_data(n, p, m, x_tr, y_tr, w0, edges, weights,
             x_tr_, y_tr_, w0_, edges_, weights_);
    wt = malloc(sizeof(double) * (p + 1));
    losses = malloc(sizeof(double) * max_iter);
    double run_time_head = 0.0, run_time_tail = 0.0, total_time = 0.0;
    if (sparsity > p) {
        printf("the parameter of s is too large!\n");
        printf("s is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    Array *f_nodes = malloc(sizeof(Array));
    Array *f_edges = malloc(sizeof(Array));
    f_nodes->array = malloc(sizeof(int) * p);
    f_edges->array = malloc(sizeof(int) * p);
    algo_batch_graph_iht_logit(
            edges, weights, g, sparsity, p, m, x_tr, y_tr, w0, n, tol,
            max_iter, lr, eta, verbose, wt, f_nodes, losses, &run_time_head,
            &run_time_tail, &total_time);
    PyObject *results = batch_get_result(p, max_iter, total_time, wt, losses);
    free(x_tr), free(y_tr), free(w0), free(edges), free(weights);
    free(wt), free(losses), free(f_nodes->array), free(f_edges->array);
    free(f_nodes), free(f_edges);
    return results;
}


static PyObject *test(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    double sum = 0.0;
    PyArrayObject *x_tr_;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x_tr_)) { return NULL; }
    int n = (int) (x_tr_->dimensions[0]);     // number of samples
    int p = (int) (x_tr_->dimensions[1]);     // number of features
    printf("%d %d\n", n, p);
    double *x_tr = PyArray_DATA(x_tr_);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%.2f ", x_tr[i * p + j]);
            sum += x_tr[i * p + j];
        }
        printf("\n");
    }
    PyObject *results = PyFloat_FromDouble(sum);
    return results;
}


static PyMethodDef sparse_methods[] = {
        {"test",                     (PyCFunction) test,
                METH_VARARGS, "test docs"},
        {"wrap_head_tail_binsearch", (PyCFunction) wrap_head_tail_binsearch,
                METH_VARARGS, "wrap_head_tail_binsearch docs"},
        {"algo_sto_iht",             (PyCFunction) algo_sto_iht,
                METH_VARARGS, "algo_sto_iht docs"},
        {"algo_graph_sto_iht",       (PyCFunction) algo_graph_sto_iht,
                METH_VARARGS, "algo_graph_sto_iht docs"},
        {"algo_best_subset",         (PyCFunction) algo_best_subset,
                METH_VARARGS, "algo_best_subset docs"},
        {NULL, NULL, 0, NULL}};

/** Python version 2 for module initialization */
PyMODINIT_FUNC initsparse_module() {
    Py_InitModule3("sparse_module", sparse_methods,
                   "some docs for sparse learning algorithms.");
    import_array();
}

int main() {
    printf("test of main wrapper!\n");
}