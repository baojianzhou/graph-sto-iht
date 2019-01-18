#include <Python.h>
#include <numpy/arrayobject.h>
#include "head_tail_proj.h"

static PyObject *test(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("error: unknown error !!\n");
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

static PyObject *wrap_head_tail_bisearch(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    head_tail_bisearch_para *para = malloc(sizeof(head_tail_bisearch_para));
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
    head_tail_bisearch(
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


static PyMethodDef sparse_methods[] = {
        {"just a simple test function!", (PyCFunction) test,
                METH_VARARGS, "test docs"},
        {"wrap_head_tail_bisearch",     (PyCFunction) wrap_head_tail_bisearch,
                METH_VARARGS, "wrap_head_tail_bisearch docs"},
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