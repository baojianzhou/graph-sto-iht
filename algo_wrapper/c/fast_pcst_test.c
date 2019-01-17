#include "fast_pcst.h"

int comp(const void *a, const void *b) {
    return (*(int *) a - *(int *) b);
}


typedef struct {
    int n;
    int m;
    int root;
    double *costs;
    EdgePair *edges;
    double *prizes;
} Data;

void print_result(Array *result_nodes, Array *result_edges, int number) {
    qsort(result_edges->array, (size_t) result_edges->size, sizeof(int), comp);
    qsort(result_nodes->array, (size_t) result_nodes->size, sizeof(int), comp);
    printf(" ------------------- test %d-result ------------------- \n",
           number);
    printf("nodes: ");
    for (int i = 0; i < result_nodes->size; i++) {
        printf(" %d", result_nodes->array[i]);
    }
    printf("\nedges: ");
    for (int i = 0; i < result_edges->size; i++) {
        printf(" %d", result_edges->array[i]);
    }
    printf("\n");
}

int
_is_passed(Array *result_nodes, Array *result_edges, const int *_result_nodes,
           const int *_result_edges, int n_, int m_) {
    for (int i = 0; i < result_nodes->size; i++) {
        if (result_nodes->array[i] != _result_nodes[i]) {
            return 0;
        }
    }
    if (result_nodes->size != n_) {
        return 0;
    }
    if (result_edges->size != m_) {
        return 0;
    }
    for (int i = 0; i < result_edges->size; i++) {
        if (result_edges->array[i] != _result_edges[i]) {
            return 0;
        }
    }
    printf("test passed!\n");
    return 1;
}

int test_1() {
    int root = 0, target_num_active_clusters = 0, n = 3, m = 2, verbose = 0;
    PruningMethod pruning = NoPruning;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->array = malloc(sizeof(int) * n);
    int _result_nodes[] = {0, 1, 2}, _result_edges[] = {0, 1};;
    int edge_u[] = {0, 1};
    int edge_v[] = {1, 2};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    double prizes[] = {0., 5., 6.};
    double costs[] = {3., 4.};
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 1);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      3, 2);
}

int test_2() {
    int root = -1, target_num_active_clusters = 1, n = 3, m = 2, verbose = 0;
    PruningMethod pruning = NoPruning;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->array = malloc(sizeof(int) * n);
    int _result_nodes[] = {1, 2}, _result_edges[] = {1};;
    int edge_u[] = {0, 1}, edge_v[] = {1, 2};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    double prizes[] = {0., 5., 6.};
    double costs[] = {3., 4.};
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 2);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      2, 1);
}

int test_3() {
    int root = -1, target_num_active_clusters = 1, n = 3, m = 2, verbose = 0;
    PruningMethod pruning = GWPruning;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->array = malloc(sizeof(int) * n);
    int _result_nodes[] = {1, 2}, _result_edges[] = {1};;
    int edge_u[] = {0, 1}, edge_v[] = {1, 2};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    double prizes[] = {0., 5., 6.};
    double costs[] = {3., 4.};
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 3);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      2, 1);
}

int test_4() {
    int root = -1, target_num_active_clusters = 1, n = 3, m = 2, verbose = 0;
    PruningMethod pruning = StrongPruning;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->array = malloc(sizeof(int) * n);
    int edge_u[] = {0, 1}, edge_v[] = {1, 2};
    int _result_nodes[] = {1, 2}, _result_edges[] = {1};;
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    double prizes[] = {0., 5., 6.};
    double costs[] = {3., 4.};
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 4);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      2, 1);
}

int test_5() {
    int root = 0, target_num_active_clusters = 0, n = 4, m = 3, verbose = 0;
    PruningMethod pruning = NoPruning;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->array = malloc(sizeof(int) * 4);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->array = malloc(sizeof(int) * 4);
    int edge_u[] = {0, 1, 2}, edge_v[] = {1, 2, 3};
    int _result_nodes[] = {0, 1, 2, 3}, _result_edges[] = {1, 2};;
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    double prizes[] = {10., 0., 1., 10.};
    double costs[] = {10., 4., 3.};
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 5);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      4, 2);
}

int test_6() {
    int root = 0, target_num_active_clusters = 0, n = 4, m = 3, verbose = 0;
    PruningMethod pruning = GWPruning;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->array = malloc(sizeof(int) * 4);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->array = malloc(sizeof(int) * 4);
    int edge_u[] = {0, 1, 2};
    int edge_v[] = {1, 2, 3};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    int _result_nodes[] = {0}, _result_edges[] = {};
    double prizes[] = {10., 0., 1., 10.};
    double costs[] = {10., 4., 3.};
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 6);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      1, 0);
}

int test_7() {
    int root = 0, target_num_active_clusters = 0, n = 4, m = 3, verbose = 0;
    PruningMethod pruning = NoPruning;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->array = malloc(sizeof(int) * n);
    int edge_u[] = {0, 1, 2};
    int edge_v[] = {1, 2, 3};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    int _result_nodes[] = {0, 1, 2, 3}, _result_edges[] = {0, 1, 2};
    double prizes[] = {10., 10., 1., 10.};
    double costs[] = {10., 6., 5.};
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 7);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      4, 3);
}

int test_8() {
    int root = 0, target_num_active_clusters = 0, n = 4, m = 3, verbose = 0;
    PruningMethod pruning = GWPruning;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->array = malloc(sizeof(int) * n);
    int edge_u[] = {0, 1, 2};
    int edge_v[] = {1, 2, 3};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    int _result_nodes[] = {0, 1, 2, 3}, _result_edges[] = {0, 1, 2};
    double prizes[] = {10., 10., 1., 10.};
    double costs[] = {10., 6., 5.};
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 8);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      4, 3);
}

int test_9() {
    int root = 0, target_num_active_clusters = 0, n = 3, m = 2, verbose = 0;
    PruningMethod pruning = NoPruning;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->array = malloc(sizeof(int) * 4);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->array = malloc(sizeof(int) * 4);
    int edge_u[] = {0, 1};
    int edge_v[] = {1, 2};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    int _result_nodes[] = {0, 1, 2}, _result_edges[] = {1};
    double prizes[] = {10., 3., 3.};
    double costs[] = {100., 2.};
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 9);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      3, 1);
}

int test_10() {
    int root = 0, target_num_active_clusters = 0, n = 3, m = 2, verbose = 0;
    PruningMethod pruning = GWPruning;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->array = malloc(sizeof(int) * n);
    int edge_u[] = {0, 1};
    int edge_v[] = {1, 2};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    int _result_nodes[] = {0}, _result_edges[] = {};
    double prizes[] = {10., 3., 3.};
    double costs[] = {100., 2.};
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 10);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      1, 0);
}

int test_11() {
    int root = -1, target_num_active_clusters = 2, n = 3, m = 2, verbose = 0;
    PruningMethod pruning = GWPruning;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->array = malloc(sizeof(int) * n);
    int edge_u[] = {0, 1};
    int edge_v[] = {1, 2};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    int _result_nodes[] = {0, 1, 2}, _result_edges[] = {1};
    double prizes[] = {10., 3., 3.};
    double costs[] = {100., 2.};
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 11);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      3, 1);
}

int test_12() {
    int root = -1, target_num_active_clusters = 1, n = 3, m = 2, verbose = 0;
    PruningMethod pruning = GWPruning;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->array = malloc(sizeof(int) * n);
    int edge_u[] = {0, 1};
    int edge_v[] = {1, 2};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    int _result_nodes[] = {0}, _result_edges[] = {};
    double prizes[] = {10., 3., 3.};
    double costs[] = {100., 2.};
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 12);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      1, 0);
}

int test_13() {
    int root = -1, target_num_active_clusters = 2, n = 4, m = 3, verbose = 0;
    PruningMethod pruning = GWPruning;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->array = malloc(sizeof(int) * n);
    int edge_u[] = {0, 1, 2};
    int edge_v[] = {1, 2, 3};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    int _result_nodes[] = {0, 2, 3}, _result_edges[] = {2};
    double prizes[] = {10., 0., 6., 6.};
    double costs[] = {100., 2., 5.};
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 13);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      3, 1);
}

int test_14() {
    int i, root = 3, target_num_active_clusters = 0;
    int n = 10, m = 24, verbose = 0;
    int edge_u[] = {0, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4,
                    4, 4, 5, 6};
    int edge_v[] = {1, 2, 3, 9, 2, 3, 5, 9, 3, 5, 7, 8, 3, 4, 5, 6, 7, 8, 9, 5,
                    6, 7, 8, 8};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    int _result_nodes[] = {3, 4, 6, 7, 8}, _result_edges[] = {16, 20, 21, 23};
    double prizes[] = {0.032052554364677466, 0.32473378289799926,
                       0.069699345546302638, 0,
                       0.74867253235151754, 0.19804330340026255,
                       0.85430521133171622, 0.83819939651391351,
                       0.71744625276884877, 0.016798567754083948};
    double costs[] = {0.8, 0.8, 0.8800000000000001, 0.8, 0.8,
                      0.8800000000000001, 0.8,
                      0.8, 0.8800000000000001, 0.8, 0.8, 0.8,
                      0.8800000000000001,
                      0.8800000000000001, 0.8800000000000001,
                      0.8800000000000001, 0.8800000000000001,
                      0.8800000000000001, 0.8800000000000001, 0.8, 0.8, 0.8,
                      0.8, 0.8};
    double eps = 1e-6;
    PruningMethod pruning = GWPruning;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->size = 0;
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->size = 0;
    result_edges->array = malloc(sizeof(int) * n);
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 14);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      5, 4);
}

int test_15() {
    int i, root = -1, target_num_active_clusters = 1, n = 8, m = 7, verbose = 0;
    int edge_u[] = {0, 1, 2, 3, 4, 5, 6};
    int edge_v[] = {1, 2, 3, 4, 5, 6, 7};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    int _result_nodes[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int _result_edges[] = {0, 1, 2, 3, 4, 5, 6};
    double prizes[] = {100., 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 100.0};
    double costs[] = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};
    double eps = 1e-6;
    PruningMethod pruning = GWPruning;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->size = 0;
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->size = 0;
    result_edges->array = malloc(sizeof(int) * n);
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 15);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      8, 7);
}

int test_16() {
    int i, root = -1, target_num_active_clusters = 1;
    int n = 5, m = 4, verbose = 0;
    int edge_u[] = {0, 0, 2, 3};
    int edge_v[] = {1, 2, 3, 4};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    int _result_nodes[] = {1}, _result_edges[] = {};
    double prizes[] = {0., 2.2, 0.0, 0.0, 2.1};
    double costs[] = {1.0, 1.0, 1.0, 1.0};
    double eps = 1e-6;
    PruningMethod pruning = StrongPruning;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->size = 0;
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->size = 0;
    result_edges->array = malloc(sizeof(int) * n);
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 16);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      1, 0);
}

int test_17() {
    int i, root = -1, target_num_active_clusters = 1;
    int n = 5, m = 4, verbose = 0;
    int edge_u[] = {0, 0, 2, 3};
    int edge_v[] = {1, 2, 3, 4};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    int _result_nodes[] = {0, 1, 2, 3, 4}, _result_edges[] = {0, 1, 2, 3};
    double prizes[] = {0., 2.2, 0.0, 0.0, 2.1};
    double costs[] = {1.0, 1.0, 1.0, 1.0};
    double eps = 1e-6;
    PruningMethod pruning = GWPruning;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->size = 0;
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->size = 0;
    result_edges->array = malloc(sizeof(int) * n);
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 17);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      5, 4);
}

int test_18() {
    int i, root = -1, target_num_active_clusters = 1;
    int n = 3, m = 2, verbose = 0;
    int edge_u[] = {0, 1};
    int edge_v[] = {1, 2};
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (i = 0; i < m; i++) {
        edges[i].first = edge_u[i];
        edges[i].second = edge_v[i];
    }
    int _result_nodes[] = {0, 1}, _result_edges[] = {0};
    double prizes[] = {2., 2., 2.};
    double costs[] = {0.0, 5.0};
    double eps = 1e-6;
    PruningMethod pruning = GWPruning;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->size = 0;
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->size = 0;
    result_edges->array = malloc(sizeof(int) * n);
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m,
                           verbose);
    run_pcst(pcst, result_nodes, result_edges);
    print_result(result_nodes, result_edges, 18);
    return _is_passed(result_nodes, result_edges, _result_nodes, _result_edges,
                      2, 1);
}

void test_on_simu() {
    int num_passed = 0;
    num_passed += test_1() + test_2() + test_3() + test_4();
    num_passed += test_5() + test_6() + test_7() + test_8();
    num_passed += test_9() + test_10() + test_11() + test_12();
    num_passed += test_13() + test_14() + test_15() + test_16();
    num_passed += test_17() + test_18();
    printf("total number of passed: %d / 18\n", num_passed);
}


int main() {
    test_on_simu();
    return (EXIT_SUCCESS);
}