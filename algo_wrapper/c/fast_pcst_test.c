//
// Created by baojian on 8/4/18.
//
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
    int i, root = 3, target_num_active_clusters = 0, n = 10, m = 24, verbose = 0;
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
    int _result_nodes[] = {0, 1, 2, 3, 4, 5, 6, 7}, _result_edges[] = {0, 1, 2,
                                                                       3, 4, 5,
                                                                       6};
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
    int i, root = -1, target_num_active_clusters = 1, n = 5, m = 4, verbose = 0;
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
    int i, root = -1, target_num_active_clusters = 1, n = 5, m = 4, verbose = 0;
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
    int i, root = -1, target_num_active_clusters = 1, n = 3, m = 2, verbose = 0;
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

void read_stp_file(char *file_name, Data *graph) {
    FILE *fp;
    long line_len;
    int n = 0, m = 0, root = -1;
    EdgePair *edges = malloc(sizeof(EdgePair));
    double *prizes = malloc(sizeof(double)), *costs = malloc(sizeof(double));
    char *line = NULL;
    char tokens[30][100];
    size_t len = 0, num_lines = 0, edges_size = 0, prize_size = 0;
    if ((fp = fopen(file_name, "r")) == NULL) {
        printf("cannot open: %s!\n", file_name);
        exit(EXIT_FAILURE);
    }
    while ((line_len = getline(&line, &len, fp)) != -1) {
        for (int k = 0; k < line_len; k++) {
            if (line[k] == '\t') {
                line[k] = ' ';
            }
        }
        int tokens_size = 0;
        for (char *token = strtok(line, " ");
             token != NULL; token = strtok(NULL, " ")) {
            strcpy(tokens[tokens_size++], token);
        }
        for (int i = 0; i < tokens_size; i++) {
            if (strcmp("Nodes", tokens[i]) == 0) {
                n = (int) strtol(tokens[i + 1], NULL, 10);
                prizes = malloc(sizeof(double) * n);
                for (int k = 0; k < n; k++) {
                    prizes[k] = 0.0;
                }
            }
            if (strcmp("Edges", tokens[i]) == 0) {
                m = (int) strtol(tokens[i + 1], NULL, 10);
                edges = malloc(sizeof(EdgePair) * m);
                costs = malloc(sizeof(double) * m);
            }
            if (strcmp("E", tokens[i]) == 0) {
                edges[edges_size].first =
                        (int) strtol(tokens[i + 1], NULL, 10) - 1;
                costs[edges_size] = strtod(tokens[i + 3], NULL);
                edges[edges_size++].second =
                        (int) strtol(tokens[i + 2], NULL, 10) - 1;
            }
            if ((strcmp("TP", tokens[i]) == 0) ||
                strcmp("T", tokens[i]) == 0) {
                int index = (int) strtol(tokens[i + 1], NULL, 10) - 1;
                double tmp = strtod(tokens[i + 2], NULL);
                if (tmp < 0.0) {
                    tmp = -tmp;
                }
                prizes[index] = tmp;
                prize_size++;
            }
            if (strcmp("RootP", tokens[i]) == 0) {
                root = (int) strtol(tokens[i + 1], NULL, 10) - 1;
            }
        }
        num_lines++;
    }
    fclose(fp);
    graph->n = n;
    graph->m = m;
    graph->root = root;
    graph->costs = costs;
    graph->edges = edges;
    graph->prizes = prizes;
}

void test_stp_instance(char *file_name, char *index, int *n_, int *m_,
                       double *run_time_, int *re_n, int *re_m,
                       PruningMethod pruning, int target_num_active_clusters) {
    EdgePair *edges;
    double *prizes, *costs;
    Data *graph = malloc(sizeof(Data));
    read_stp_file(file_name, graph);
    int n, m, root = graph->root;
    if (root >= 0) {
        target_num_active_clusters = 0;
    }
    n = graph->n;
    m = graph->m;
    prizes = graph->prizes;
    costs = graph->costs;
    edges = graph->edges;
    double eps = 1e-6;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->size = 0;
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->size = 0;
    result_edges->array = malloc(sizeof(int) * n);
    clock_t begin = clock();
    //allocate all of the memory
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           target_num_active_clusters, eps, pruning, n, m, 0);
    run_pcst(pcst, result_nodes, result_edges);
    free_pcst(pcst);
    clock_t end = clock();
    double time_spent = (double) (end - begin) / 1000.;
    qsort(result_edges->array, (size_t) result_edges->size, sizeof(int), comp);
    qsort(result_nodes->array, (size_t) result_nodes->size, sizeof(int), comp);
    printf("id:%s number of nodes: %d number of edges: %d, runt_time:%.6f\n",
           index, result_nodes->size, result_edges->size, time_spent);
    *n_ = n;
    *m_ = m;
    *run_time_ = time_spent;
    *re_n = result_nodes->size;
    *re_m = result_edges->size;
}

/**
 * Table 1
 * maximal buffer of pairing heap used is 1045
 */
void test_table_1_pcspg_actmodpc() {
    int num_cases = 8;
    int num_nodes[num_cases], num_edges[num_cases], re_n[num_cases], re_m[num_cases];
    char file_name[100], *root_path = "/network/rit/lab/ceashpc/bz383376/data/pcst/PCSPG-ACTMODPC/";
    char *file_list[] = {"HCMV", "drosophila001", "drosophila005",
                         "drosophila0075",
                         "lymphoma", "metabol_expr_mice_1",
                         "metabol_expr_mice_2", "metabol_expr_mice_3"};
    double run_times[num_cases];
    PruningMethod pruning = GWPruning;
    for (int i = 0; i < num_cases; i++) {
        strcpy(file_name, root_path);
        strcat(file_name, file_list[i]);
        strcat(file_name, ".stp");
        test_stp_instance(file_name, file_list[i], &num_nodes[i],
                          &num_edges[i],
                          &run_times[i], &re_n[i], &re_m[i], pruning, 1);
    }
    printf("\n");
    printf("-------- Table 1 Results for the PCSPG-ACTMODPC test instances --------\n");
    printf("-----------------------------------------------------------------------\n");
    printf("Instance                 n       m     re_n      re_m     Time(ms)     \n");
    printf("-----------------------------------------------------------------------\n");
    for (int i = 0; i < num_cases; i++) {
        printf("%-19s    %4d    %5d    %4d    %5d    %.3f\n",
               file_list[i], num_nodes[i], num_edges[i], re_n[i], re_m[i],
               run_times[i]);
    }
}

/**
 * Table 2
 */
void test_table_2_pcspg_h() {
    int num_cases = 14;
    int num_nodes[num_cases], num_edges[num_cases], re_n[num_cases], re_m[num_cases];
    char file_name[100], *header = "/network/rit/lab/ceashpc/bz383376/data/pcst/PCSPG-H/";
    char *file_list[] = {"hc10p", "hc10u", "hc11p", "hc11u", "hc12p", "hc12u",
                         "hc6p", "hc6u", "hc7p", "hc7u", "hc8p", "hc8u",
                         "hc9p", "hc9u"};
    double run_times[num_cases];
    PruningMethod pruning = GWPruning;
    for (int i = 0; i < num_cases; i++) {
        strcpy(file_name, header);
        strcat(file_name, file_list[i]);
        strcat(file_name, ".stp");
        test_stp_instance(file_name, file_list[i], &num_nodes[i],
                          &num_edges[i],
                          &run_times[i], &re_n[i], &re_m[i], pruning, 1);
    }
    printf("\n");
    printf("----------- Table 2 Results for the PCSPG-H test instances ------------\n");
    printf("-----------------------------------------------------------------------\n");
    printf("Instance                 n       m     re_n      re_m     Time(ms)     \n");
    printf("-----------------------------------------------------------------------\n");
    for (int i = 0; i < num_cases; i++) {
        printf("%-19s    %4d    %5d    %4d    %5d    %.3f\n",
               file_list[i], num_nodes[i], num_edges[i], re_n[i], re_m[i],
               run_times[i]);
    }
}


/**
 * Table 3 of http://people.csail.mit.edu/ludwigs/papers/dimacs14_fastpcst.pdf
 */
void test_table_3_pcspg_h2() {
    int num_cases = 14;
    char file_name[100], *header = "/network/rit/lab/ceashpc/bz383376/data/pcst/PCSPG-H2/";
    char *file_list[] = {"hc10p2", "hc10u2", "hc11p2", "hc11u2", "hc12p2",
                         "hc12u2",
                         "hc6p2", "hc6u2", "hc7p2", "hc7u2", "hc8p2", "hc8u2",
                         "hc9p2", "hc9u2"};
    int num_nodes[num_cases], num_edges[num_cases], re_n[num_cases], re_m[num_cases];
    double run_times[num_cases];
    PruningMethod pruning = GWPruning;
    for (int i = 0; i < num_cases; i++) {
        strcpy(file_name, header);
        strcat(file_name, file_list[i]);
        strcat(file_name, ".stp");
        test_stp_instance(file_name, file_list[i], &num_nodes[i],
                          &num_edges[i],
                          &run_times[i], &re_n[i], &re_m[i], pruning, 1);
    }
    printf("\n");
    printf("------------ Table 3 Results for the PCSPG-H2 test instances ----------\n");
    printf("-----------------------------------------------------------------------\n");
    printf("Instance       n       m     re_n      re_m     Time(ms)     \n");
    printf("-----------------------------------------------------------------------\n");
    for (int i = 0; i < num_cases; i++) {
        printf("%-10s    %4d    %5d    %4d    %5d    %.3f\n",
               file_list[i], num_nodes[i], num_edges[i], re_n[i], re_m[i],
               run_times[i]);
    }
}

/**
 * Table 4
 */
void test_table_4_pcspg_hand() {
    int num_cases = 48;
    int num_nodes[num_cases], num_edges[num_cases], re_n[num_cases], re_m[num_cases];
    char file_name[100], *root_path = "/network/rit/lab/ceashpc/bz383376/data/pcst/PCSPG-hand/";
    char *file_list[] = {"handbd01", "handbd02", "handbd03", "handbd04",
                         "handbd05", "handbd06", "handbd07",
                         "handbd08", "handbd09", "handbd10", "handbd11",
                         "handbd12", "handbd13", "handbd14",
                         "handbi01", "handbi02", "handbi03", "handbi04",
                         "handbi05", "handbi06", "handbi07",
                         "handbi08", "handbi09", "handbi10", "handbi11",
                         "handbi12", "handbi13", "handbi14",
                         "handsd01", "handsd02", "handsd03", "handsd04",
                         "handsd05", "handsd06", "handsd07",
                         "handsd08", "handsd09", "handsd10", "handsi01",
                         "handsi02", "handsi03", "handsi04",
                         "handsi05", "handsi06", "handsi07", "handsi08",
                         "handsi09", "handsi10"};
    double run_times[num_cases];
    PruningMethod pruning = GWPruning;
    for (int i = 0; i < num_cases; i++) {
        strcpy(file_name, root_path);
        strcat(file_name, file_list[i]);
        strcat(file_name, ".stp");
        test_stp_instance(file_name, file_list[i], &num_nodes[i],
                          &num_edges[i],
                          &run_times[i], &re_n[i], &re_m[i], pruning, 1);
    }
    printf("\n");
    printf("------------ Table 4 Results for the PCSPG-H2 test instances ----------\n");
    printf("-----------------------------------------------------------------------\n");
    printf("Instance       n         m       re_n      re_m     Time(ms)     \n");
    printf("-----------------------------------------------------------------------\n");
    for (int i = 0; i < num_cases; i++) {
        printf("%-10s    %6d    %6d    %5d    %5d    %.3f\n",
               file_list[i], num_nodes[i], num_edges[i], re_n[i], re_m[i],
               run_times[i]);
    }
}

void test_table_5_pcspg_crr() {
    int num_cases = 80;
    int num_nodes[num_cases], num_edges[num_cases], re_n[num_cases], re_m[num_cases];
    char file_name[100], *root_path = "/network/rit/lab/ceashpc/bz383376/data/pcst/PCSPG-CRR/";
    char *file_list[] = {"C01-A", "C01-B", "C02-A", "C02-B", "C03-A", "C03-B",
                         "C04-A", "C04-B", "C05-A", "C05-B",
                         "C06-A", "C06-B", "C07-A", "C07-B", "C08-A", "C08-B",
                         "C09-A", "C09-B", "C10-A", "C10-B",
                         "C11-A", "C11-B", "C12-A", "C12-B", "C13-A", "C13-B",
                         "C14-A", "C14-B", "C15-A", "C15-B",
                         "C16-A", "C16-B", "C17-A", "C17-B", "C18-A", "C18-B",
                         "C19-A", "C19-B", "C20-A", "C20-B",
                         "D01-A", "D01-B", "D02-A", "D02-B", "D03-A", "D03-B",
                         "D04-A", "D04-B", "D05-A", "D05-B",
                         "D06-A", "D06-B", "D07-A", "D07-B", "D08-A", "D08-B",
                         "D09-A", "D09-B", "D10-A", "D10-B",
                         "D11-A", "D11-B", "D12-A", "D12-B", "D13-A", "D13-B",
                         "D14-A", "D14-B", "D15-A", "D15-B",
                         "D16-A", "D16-B", "D17-A", "D17-B", "D18-A", "D18-B",
                         "D19-A", "D19-B", "D20-A", "D20-B"};
    double run_times[num_cases];
    PruningMethod pruning = GWPruning;
    for (int i = 0; i < num_cases; i++) {
        strcpy(file_name, root_path);
        strcat(file_name, file_list[i]);
        strcat(file_name, ".stp");
        test_stp_instance(file_name, file_list[i], &num_nodes[i],
                          &num_edges[i],
                          &run_times[i], &re_n[i], &re_m[i], pruning, 1);
    }
    printf("\n");
    printf("------------ Table 5 Results for the PCSPG-H2 test instances ----------\n");
    printf("-----------------------------------------------------------------------\n");
    printf("Instance       n         m       re_n      re_m     Time(ms)           \n");
    printf("-----------------------------------------------------------------------\n");
    for (int i = 0; i < num_cases; i++) {
        printf("%-10s    %6d    %6d    %5d    %5d    %.3f\n",
               file_list[i], num_nodes[i], num_edges[i], re_n[i], re_m[i],
               run_times[i]);
    }
}

void test_table_6_pcspg_i640() {
    int num_cases = 100;
    int num_nodes[num_cases], num_edges[num_cases], re_n[num_cases], re_m[num_cases];
    char file_name[100], *root_path = "/network/rit/lab/ceashpc/bz383376/data/pcst/PCSPG-i640/";
    char *file_list[] = {"i640-001", "i640-002", "i640-003", "i640-004",
                         "i640-005",
                         "i640-011", "i640-012", "i640-013", "i640-014",
                         "i640-015",
                         "i640-021", "i640-022", "i640-023", "i640-024",
                         "i640-025",
                         "i640-031", "i640-032", "i640-033", "i640-034",
                         "i640-035",
                         "i640-041", "i640-042", "i640-043", "i640-044",
                         "i640-045",
                         "i640-101", "i640-102", "i640-103", "i640-104",
                         "i640-105",
                         "i640-111", "i640-112", "i640-113", "i640-114",
                         "i640-115",
                         "i640-121", "i640-122", "i640-123", "i640-124",
                         "i640-125",
                         "i640-131", "i640-132", "i640-133", "i640-134",
                         "i640-135",
                         "i640-141", "i640-142", "i640-143", "i640-144",
                         "i640-145",
                         "i640-201", "i640-202", "i640-203", "i640-204",
                         "i640-205",
                         "i640-211", "i640-212", "i640-213", "i640-214",
                         "i640-215",
                         "i640-221", "i640-222", "i640-223", "i640-224",
                         "i640-225",
                         "i640-231", "i640-232", "i640-233", "i640-234",
                         "i640-235",
                         "i640-241", "i640-242", "i640-243", "i640-244",
                         "i640-245",
                         "i640-301", "i640-302", "i640-303", "i640-304",
                         "i640-305",
                         "i640-311", "i640-312", "i640-313", "i640-314",
                         "i640-315",
                         "i640-321", "i640-322", "i640-323", "i640-324",
                         "i640-325",
                         "i640-331", "i640-332", "i640-333", "i640-334",
                         "i640-335",
                         "i640-341", "i640-342", "i640-343", "i640-344",
                         "i640-345"};
    double run_times[num_cases];
    PruningMethod pruning = GWPruning;
    for (int i = 0; i < num_cases; i++) {
        strcpy(file_name, root_path);
        strcat(file_name, file_list[i]);
        strcat(file_name, ".stp");
        test_stp_instance(file_name, file_list[i], &num_nodes[i],
                          &num_edges[i],
                          &run_times[i], &re_n[i], &re_m[i], pruning, 1);
    }
    printf("\n");
    printf("------------ Table 6 Results for the PCSPG-H2 test instances ----------\n");
    printf("-----------------------------------------------------------------------\n");
    printf("Instance    n         m       re_n      re_m     Time(ms)     \n");
    printf("-----------------------------------------------------------------------\n");
    for (int i = 0; i < num_cases; i++) {
        printf("%-10s    %6d    %6d    %5d    %5d    %.3f\n",
               file_list[i], num_nodes[i], num_edges[i], re_n[i], re_m[i],
               run_times[i]);
    }
}

void test_table_7_pcspg_jmp() {
    int num_cases = 34;
    int num_nodes[num_cases], num_edges[num_cases], re_n[num_cases], re_m[num_cases];
    char file_name[100], *root_path = "/network/rit/lab/ceashpc/bz383376/data/pcst/PCSPG-JMP/";
    char *file_list[] = {"K100", "K100.1", "K100.10", "K100.2", "K100.3",
                         "K100.4", "K100.5", "K100.6",
                         "K100.7", "K100.8", "K100.9", "K200", "K400",
                         "K400.1", "K400.10", "K400.2",
                         "K400.3", "K400.4", "K400.5", "K400.6", "K400.7",
                         "K400.8", "K400.9",
                         "P100", "P100.1", "P100.2", "P100.3", "P100.4",
                         "P200", "P400", "P400.1",
                         "P400.2", "P400.3", "P400.4"};
    double run_times[num_cases];
    PruningMethod pruning = GWPruning;
    for (int i = 0; i < num_cases; i++) {
        strcpy(file_name, root_path);
        strcat(file_name, file_list[i]);
        strcat(file_name, ".stp");
        test_stp_instance(file_name, file_list[i], &num_nodes[i],
                          &num_edges[i],
                          &run_times[i], &re_n[i], &re_m[i], pruning, 1);
    }
    printf("\n");
    printf("------------ Table 6 Results for the PCSPG-H2 test instances ----------\n");
    printf("-----------------------------------------------------------------------\n");
    printf("Instance    n         m       re_n      re_m     Time(ms)     \n");
    printf("-----------------------------------------------------------------------\n");
    for (int i = 0; i < num_cases; i++) {
        printf("%-10s    %6d    %6d    %5d    %5d    %.3f\n",
               file_list[i], num_nodes[i], num_edges[i], re_n[i], re_m[i],
               run_times[i]);
    }
}

void test_table_8_pcspg_pucnu() {
    int num_cases = 18;
    int num_nodes[num_cases], num_edges[num_cases], re_n[num_cases], re_m[num_cases];
    char file_name[100], *root_path = "/network/rit/lab/ceashpc/bz383376/data/pcst/PCSPG-PUCNU/";
    char *file_list[] = {"bip42nu", "bip52nu", "bip62nu", "bipa2nu", "bipe2nu",
                         "cc10-2nu", "cc11-2nu", "cc12-2nu", "cc3-10nu",
                         "cc3-11nu", "cc3-12nu",
                         "cc3-4nu", "cc3-5nu", "cc5-3nu", "cc6-2nu", "cc6-3nu",
                         "cc7-3nu", "cc9-2nu"};
    double run_times[num_cases];
    PruningMethod pruning = GWPruning;
    for (int i = 0; i < num_cases; i++) {
        strcpy(file_name, root_path);
        strcat(file_name, file_list[i]);
        strcat(file_name, ".stp");
        test_stp_instance(file_name, file_list[i], &num_nodes[i],
                          &num_edges[i],
                          &run_times[i], &re_n[i], &re_m[i], pruning, 1);
    }
    printf("\n");
    printf("------------ Table 6 Results for the PCSPG-H2 test instances ----------\n");
    printf("-----------------------------------------------------------------------\n");
    printf("Instance      n         m       re_n      re_m     Time(ms)     \n");
    printf("-----------------------------------------------------------------------\n");
    for (int i = 0; i < num_cases; i++) {
        printf("%-8s    %6d    %6d    %5d    %5d    %.3f\n",
               file_list[i], num_nodes[i], num_edges[i], re_n[i], re_m[i],
               run_times[i]);
    }
}

void test_table_9_pcspg_random() {
    int num_cases = 68;
    int num_nodes[num_cases], num_edges[num_cases], re_n[num_cases], re_m[num_cases];
    char file_name[100], *root_path = "/network/rit/lab/ceashpc/bz383376/data/pcst/PCSPG-RANDOM/";
    char *file_list[] = {"a0200RandGraph.1.2", "a10000RandGraph.1.5",
                         "a14000RandGraph.2", "a2000RandGraph.3",
                         "a0200RandGraph.1.5", "a10000RandGraph.2",
                         "a14000RandGraph.3", "a3000RandGraph.1.2",
                         "a0200RandGraph.2", "a10000RandGraph.3",
                         "a1400RandGraph.1.2", "a3000RandGraph.1.5",
                         "a0200RandGraph.3", "a1000RandGraph.1.2",
                         "a1400RandGraph.1.5", "a3000RandGraph.2",
                         "a0400RandGraph.1.2", "a1000RandGraph.1.5",
                         "a1400RandGraph.2", "a3000RandGraph.3",
                         "a0400RandGraph.1.5", "a1000RandGraph.2",
                         "a1400RandGraph.3", "a4000RandGraph.1.2",
                         "a0400RandGraph.2", "a1000RandGraph.3",
                         "a1600RandGraph.1.2", "a4000RandGraph.1.5",
                         "a0400RandGraph.3", "a12000RandGraph.1.2",
                         "a1600RandGraph.1.5", "a4000RandGraph.2",
                         "a0600RandGraph.1.2", "a12000RandGraph.1.5",
                         "a1600RandGraph.2", "a4000RandGraph.3",
                         "a0600RandGraph.1.5", "a12000RandGraph.2",
                         "a1600RandGraph.3", "a6000RandGraph.1.2",
                         "a0600RandGraph.2", "a12000RandGraph.3",
                         "a1800RandGraph.1.2", "a6000RandGraph.1.5",
                         "a0600RandGraph.3", "a1200RandGraph.1.2",
                         "a1800RandGraph.1.5", "a6000RandGraph.2",
                         "a0800RandGraph.1.2", "a1200RandGraph.1.5",
                         "a1800RandGraph.2", "a6000RandGraph.3",
                         "a0800RandGraph.1.5", "a1200RandGraph.2",
                         "a1800RandGraph.3", "a8000RandGraph.1.2",
                         "a0800RandGraph.2", "a1200RandGraph.3",
                         "a2000RandGraph.1.2", "a8000RandGraph.1.5",
                         "a0800RandGraph.3", "a14000RandGraph.1.2",
                         "a2000RandGraph.1.5", "a8000RandGraph.2",
                         "a10000RandGraph.1.2", "a14000RandGraph.1.5",
                         "a2000RandGraph.2", "a8000RandGraph.3"};
    double run_times[num_cases];
    PruningMethod pruning = GWPruning;
    for (int i = 0; i < num_cases; i++) {
        strcpy(file_name, root_path);
        strcat(file_name, file_list[i]);
        strcat(file_name, ".stp");
        test_stp_instance(file_name, file_list[i], &num_nodes[i],
                          &num_edges[i],
                          &run_times[i], &re_n[i], &re_m[i], pruning, 1);
    }
    printf("\n");
    printf("------------ Table 6 Results for the PCSPG-Random test instances ----------\n");
    printf("-----------------------------------------------------------------------\n");
    printf("Instance      n         m       re_n      re_m     Time(ms)     \n");
    printf("-----------------------------------------------------------------------\n");
    for (int i = 0; i < num_cases; i++) {
        printf("%-25s    %6d    %6d    %5d    %5d    %.3f\n",
               file_list[i], num_nodes[i], num_edges[i], re_n[i], re_m[i],
               run_times[i]);
    }
}

void test_table_10_rpcst_cologne() {
    int num_cases = 29;
    int num_nodes[num_cases], num_edges[num_cases], re_n[num_cases], re_m[num_cases];
    char file_name[100], *root_path = "/network/rit/lab/ceashpc/bz383376/data/pcst/RPCST-cologne/";
    char *file_list[] = {"i101M1", "i101M2", "i101M3", "i102M1", "i102M2",
                         "i102M3",
                         "i103M1", "i103M2", "i103M3", "i104M2", "i104M3",
                         "i105M1", "i105M2", "i102M3",
                         "i201M2", "i201M3", "i201M4", "i202M2", "i202M3",
                         "i202M4",
                         "i203M2", "i203M3", "i203M4", "i204M2", "i204M3",
                         "i204M4", "i205M2", "i205M3", "i202M4"};
    double run_times[num_cases];
    PruningMethod pruning = GWPruning;
    for (int i = 0; i < num_cases; i++) {
        strcpy(file_name, root_path);
        strcat(file_name, file_list[i]);
        strcat(file_name, ".stp");
        test_stp_instance(file_name, file_list[i], &num_nodes[i],
                          &num_edges[i],
                          &run_times[i], &re_n[i], &re_m[i], pruning, 1);
    }
    printf("\n");
    printf("------------ Table 10 Results for the PCSPG-H2 test instances ----------\n");
    printf("------------------------------------------------------------------------\n");
    printf("Instance      n         m       re_n      re_m     Time(ms)             \n");
    printf("------------------------------------------------------------------------\n");
    for (int i = 0; i < num_cases; i++) {
        printf("%-10s    %6d    %6d    %5d    %5d    %.3f\n",
               file_list[i], num_nodes[i], num_edges[i], re_n[i], re_m[i],
               run_times[i]);
    }
}

void test_table_11_mwcs_actmod() {
    int num_cases = 8;
    int num_nodes[num_cases], num_edges[num_cases], re_n[num_cases], re_m[num_cases];
    char file_name[100], *root_path = "/network/rit/lab/ceashpc/bz383376/data/pcst/MWCS-ACTMOD/";
    char *file_list[] = {"HCMV", "drosophila001", "drosophila005",
                         "drosophila0075", "lymphoma",
                         "metabol_expr_mice_1", "metabol_expr_mice_1",
                         "metabol_expr_mice_1"};
    double run_times[num_cases];
    PruningMethod pruning = StrongPruning;
    for (int i = 0; i < num_cases; i++) {
        strcpy(file_name, root_path);
        strcat(file_name, file_list[i]);
        strcat(file_name, ".stp");
        test_stp_instance(file_name, file_list[i], &num_nodes[i],
                          &num_edges[i],
                          &run_times[i], &re_n[i], &re_m[i], pruning, 1);
    }
    printf("\n");
    printf("------------ Table 11 Results for the PCSPG-H2 test instances ----------\n");
    printf("------------------------------------------------------------------------\n");
    printf("Instance      n         m       re_n      re_m     Time(ms)             \n");
    printf("------------------------------------------------------------------------\n");
    for (int i = 0; i < num_cases; i++) {
        printf("%-20s    %6d    %6d    %5d    %5d    %.3f\n",
               file_list[i], num_nodes[i], num_edges[i], re_n[i], re_m[i],
               run_times[i]);
    }
}

void test_table_12_mwcs_jmpalmk() {
    int num_cases = 72;
    int num_nodes[num_cases], num_edges[num_cases], re_n[num_cases], re_m[num_cases];
    char file_name[100], *root_path = "/network/rit/lab/ceashpc/bz383376/data/pcst/MWCS-JMPALMK/MWCS-I-D-";
    char *file_list[] = {"n-1000-a-0.6-d-0.25-e-0.25",
                         "n-500-a-0.62-d-0.25-e-0.25",
                         "n-1000-a-0.6-d-0.25-e-0.5",
                         "n-500-a-0.62-d-0.25-e-0.5",
                         "n-1000-a-0.6-d-0.25-e-0.75",
                         "n-500-a-0.62-d-0.25-e-0.75",
                         "n-1000-a-0.6-d-0.5-e-0.25",
                         "n-500-a-0.62-d-0.5-e-0.25",
                         "n-1000-a-0.6-d-0.5-e-0.5",
                         "n-500-a-0.62-d-0.5-e-0.5",
                         "n-1000-a-0.6-d-0.5-e-0.75",
                         "n-500-a-0.62-d-0.5-e-0.75",
                         "n-1000-a-0.6-d-0.75-e-0.25",
                         "n-500-a-0.62-d-0.75-e-0.25",
                         "n-1000-a-0.6-d-0.75-e-0.5",
                         "n-500-a-0.62-d-0.75-e-0.5",
                         "n-1000-a-0.6-d-0.75-e-0.75",
                         "n-500-a-0.62-d-0.75-e-0.75",
                         "n-1000-a-1-d-0.25-e-0.25", "n-500-a-1-d-0.25-e-0.25",
                         "n-1000-a-1-d-0.25-e-0.5", "n-500-a-1-d-0.25-e-0.5",
                         "n-1000-a-1-d-0.25-e-0.75", "n-500-a-1-d-0.25-e-0.75",
                         "n-1000-a-1-d-0.5-e-0.25", "n-500-a-1-d-0.5-e-0.25",
                         "n-1000-a-1-d-0.5-e-0.5", "n-500-a-1-d-0.5-e-0.5",
                         "n-1000-a-1-d-0.5-e-0.75", "n-500-a-1-d-0.5-e-0.75",
                         "n-1000-a-1-d-0.75-e-0.25", "n-500-a-1-d-0.75-e-0.25",
                         "n-1000-a-1-d-0.75-e-0.5", "n-500-a-1-d-0.75-e-0.5",
                         "n-1000-a-1-d-0.75-e-0.75", "n-500-a-1-d-0.75-e-0.75",
                         "n-1500-a-0.6-d-0.25-e-0.25",
                         "n-750-a-0.647-d-0.25-e-0.25",
                         "n-1500-a-0.6-d-0.25-e-0.5",
                         "n-750-a-0.647-d-0.25-e-0.5",
                         "n-1500-a-0.6-d-0.25-e-0.75",
                         "n-750-a-0.647-d-0.25-e-0.75",
                         "n-1500-a-0.6-d-0.5-e-0.25",
                         "n-750-a-0.647-d-0.5-e-0.25",
                         "n-1500-a-0.6-d-0.5-e-0.5",
                         "n-750-a-0.647-d-0.5-e-0.5",
                         "n-1500-a-0.6-d-0.5-e-0.75",
                         "n-750-a-0.647-d-0.5-e-0.75",
                         "n-1500-a-0.6-d-0.75-e-0.25",
                         "n-750-a-0.647-d-0.75-e-0.25",
                         "n-1500-a-0.6-d-0.75-e-0.5",
                         "n-750-a-0.647-d-0.75-e-0.5",
                         "n-1500-a-0.6-d-0.75-e-0.75",
                         "n-750-a-0.647-d-0.75-e-0.75",
                         "n-1500-a-1-d-0.25-e-0.25", "n-750-a-1-d-0.25-e-0.25",
                         "n-1500-a-1-d-0.25-e-0.5", "n-750-a-1-d-0.25-e-0.5",
                         "n-1500-a-1-d-0.25-e-0.75", "n-750-a-1-d-0.25-e-0.75",
                         "n-1500-a-1-d-0.5-e-0.25", "n-750-a-1-d-0.5-e-0.25",
                         "n-1500-a-1-d-0.5-e-0.5", "n-750-a-1-d-0.5-e-0.5",
                         "n-1500-a-1-d-0.5-e-0.75", "n-750-a-1-d-0.5-e-0.75",
                         "n-1500-a-1-d-0.75-e-0.25", "n-750-a-1-d-0.75-e-0.25",
                         "n-1500-a-1-d-0.75-e-0.5", "n-750-a-1-d-0.75-e-0.5",
                         "n-1500-a-1-d-0.75-e-0.75",
                         "n-750-a-1-d-0.75-e-0.75"};
    double run_times[num_cases];
    PruningMethod pruning = StrongPruning;
    for (int i = 0; i < num_cases; i++) {
        strcpy(file_name, root_path);
        strcat(file_name, file_list[i]);
        strcat(file_name, ".stp");
        test_stp_instance(file_name, file_list[i], &num_nodes[i],
                          &num_edges[i], &run_times[i], &re_n[i], &re_m[i],
                          pruning, 1);
    }
    printf("\n");
    printf("------------ Table 11 Results for the PCSPG-H2 test instances ----------\n");
    printf("------------------------------------------------------------------------\n");
    printf("Instance      n         m       re_n      re_m     Time(ms)             \n");
    printf("------------------------------------------------------------------------\n");
    for (int i = 0; i < num_cases; i++) {
        printf("%-30s    %6d    %6d    %5d    %5d    %.3f\n",
               file_list[i], num_nodes[i], num_edges[i], re_n[i], re_m[i],
               run_times[i]);
    }
}

void test_all_real_cases() {
    test_table_1_pcspg_actmodpc(); //failed
    test_table_2_pcspg_h();
    test_table_3_pcspg_h2();
    test_table_4_pcspg_hand();
    test_table_5_pcspg_crr();  // failed
    test_table_6_pcspg_i640(); //failed
    test_table_7_pcspg_jmp();
    test_table_8_pcspg_pucnu();
    test_table_9_pcspg_random();
    test_table_10_rpcst_cologne();
    test_table_11_mwcs_actmod(); //failed
    test_table_12_mwcs_jmpalmk();
}


void test_on_mnist_data() {
    char *file_name = "/network/rit/lab/ceashpc/bz383376/data/pcst/MNIST/mnist_test_case_0.txt";
    int p = 784, m = 1512;
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    double *prizes = malloc(sizeof(double) * p);
    double *costs = malloc(sizeof(double) * m);
    char *line = NULL, tokens[10][20];
    FILE *fp;
    size_t len = 0, num_lines = 0, edge_index = 0;
    if ((fp = fopen(file_name, "r")) == NULL) {
        printf("cannot open: %s!\n", file_name);
        exit(EXIT_FAILURE);
    }
    printf("reading data from: %s\n", file_name);
    while ((getline(&line, &len, fp)) != -1) {
        int tokens_size = 0;
        for (char *token = strtok(line, " ");
             token != NULL; token = strtok(NULL, " ")) {
            strcpy(tokens[tokens_size++], token);
        }
        num_lines++;
        if (strcmp("E", tokens[0]) == 0) {
            int uu = (int) strtol(tokens[1], NULL, 10);
            int vv = (int) strtol(tokens[2], NULL, 10);
            double weight = strtod(tokens[3], NULL);
            edges[edge_index].first = uu;
            edges[edge_index].second = vv;
            costs[edge_index++] = weight;
            continue;
        }
        if (strcmp("N", tokens[0]) == 0) {
            int node = (int) strtol(tokens[1], NULL, 10);
            double prize = strtod(tokens[2], NULL);
            prizes[node] = prize;
            continue;
        }
    }
    fclose(fp);
    int root = -1, target_num_active_clusters = 1;
    double eps = 1e-6;
    PruningMethod pruning = GWPruning;
    int n = p;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->size = 0;
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->size = 0;
    result_edges->array = malloc(sizeof(int) * n);
    PCST *pcst;
    for (int ii = 0; ii < 10; ii++) {
        clock_t start_time = clock();
        pcst = make_pcst(edges, prizes, costs, root,
                         target_num_active_clusters, eps, pruning, n, m, 0);
        for (int kk = 0; kk < pcst->n; kk++) {
            pcst->prizes[kk] = (rand() / (RAND_MAX / (900. - 0.0)));
        }
        run_pcst(pcst, result_nodes, result_edges);
        double run_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
        printf("number of result_nodes: %d result_edges: %d run_time: %.4f\n",
               result_nodes->size, result_edges->size, run_time);
        free_pcst(pcst);
    }
}


void test_on_grid_data() {
    char *file_name = "/network/rit/lab/ceashpc/bz383376/data/icml19/text_case.txt";
    int p = 256, m = 480;
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    double *prizes = malloc(sizeof(double) * p);
    double *costs = malloc(sizeof(double) * m);
    char *line = NULL, tokens[10][20];
    FILE *fp;
    size_t len = 0, num_lines = 0, edge_index = 0;
    if ((fp = fopen(file_name, "r")) == NULL) {
        printf("cannot open: %s!\n", file_name);
        exit(EXIT_FAILURE);
    }
    printf("reading data from: %s\n", file_name);
    while ((getline(&line, &len, fp)) != -1) {
        int tokens_size = 0;
        for (char *token = strtok(line, " ");
             token != NULL; token = strtok(NULL, " ")) {
            strcpy(tokens[tokens_size++], token);
        }
        num_lines++;
        if (strcmp("E", tokens[0]) == 0) {
            int uu = (int) strtol(tokens[1], NULL, 10);
            int vv = (int) strtol(tokens[2], NULL, 10);
            double weight = strtod(tokens[3], NULL);
            edges[edge_index].first = uu;
            edges[edge_index].second = vv;
            costs[edge_index++] = weight;
            continue;
        }
        if (strcmp("N", tokens[0]) == 0) {
            int node = (int) strtol(tokens[1], NULL, 10);
            double prize = strtod(tokens[2], NULL);
            prizes[node] = prize;
            continue;
        }
    }
    fclose(fp);
    int root = -1, target_num_active_clusters = 1;
    double eps = 1e-6;
    PruningMethod pruning = GWPruning;
    int n = p;
    Array *result_nodes = malloc(sizeof(Array));
    result_nodes->size = 0;
    result_nodes->array = malloc(sizeof(int) * n);
    Array *result_edges = malloc(sizeof(Array));
    result_edges->size = 0;
    result_edges->array = malloc(sizeof(int) * n);
    for (int i = 0; i < 100; i++) {
        PCST *pcst = make_pcst(
                edges, prizes, costs, root, target_num_active_clusters, eps,
                pruning, n, m, 0);
        run_pcst(pcst, result_nodes, result_edges), free_pcst(pcst);
        printf("number of nodes: %d\n", result_nodes->size);
        printf("number of edges: %d\n", result_edges->size);
    }
}

void test_all() {
    test_on_simu();
    test_all_real_cases();
    test_on_mnist_data();
    test_on_grid_data();
}



int main() {
    test_on_simu();
    return (EXIT_SUCCESS);
}