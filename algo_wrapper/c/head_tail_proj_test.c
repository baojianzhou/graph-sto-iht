#include "head_tail_proj.h"

typedef struct {
    int p;
    int m;
    EdgePair *edges;
    double *prizes;
    double *costs;
} Data;
char *file_name = "--/mnist_test_case_0.txt";

Data *read_mnist_data() {
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
    Data *graph = malloc(sizeof(Data));
    graph->m = m;
    graph->p = p;
    graph->edges = edges;
    graph->prizes = prizes;
    graph->costs = costs;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            printf("%-6.2f ", prizes[i * 28 + j]);
        }
        printf("\n");
    }
    return graph;
}

void test_head_proj_exact() {
    Data *graph = read_mnist_data();
    PruningMethod pruning = GWPruning;
    int g = 6, sparsity = 100, max_iter = 50, root = -1, n = graph->p;
    int m = graph->m, verbose = 0;
    double C = 2 * (sparsity - 1.) * 100., delta = 1. / 169.;
    double err_tol = 1e-6, epsilon = 1e-6, run_time = 0.0;
    GraphStat *head_stat = make_graph_stat(n, m);
    head_proj_exact(
            graph->edges, graph->costs, graph->prizes, g, C, delta, max_iter,
            err_tol, root, pruning, epsilon, n, m, verbose, head_stat);
    printf("number of head_nodes: %d number of tail_nodes: %d\n",
           head_stat->re_nodes->size, head_stat->re_edges->size);
    printf("number of pcst: %d run_time: %.6f\n",
           head_stat->num_pcst, run_time);
    free_graph_stat(head_stat);
    free(graph->costs), free(graph->prizes), free(graph->edges);
    free(graph);
}

void test_head_proj_approx() {
    Data *graph = read_mnist_data();
    PruningMethod pruning = GWPruning;
    int g = 17, sparsity = 100, max_iter = 50, root = -1, n = graph->p;
    int m = graph->m, verbose = 0;
    double C = 2 * (sparsity - 1.) * graph->costs[0], delta = 1. / 169.;
    double err_tol = 1e-6, epsilon = 1e-6, run_time = 0.0;
    GraphStat *head_stat = make_graph_stat(n, m);
    head_proj_approx(
            graph->edges, graph->costs, graph->prizes, g, C, delta, max_iter,
            err_tol, root, pruning, epsilon, n, m, verbose, head_stat);
    printf("number of head_nodes: %d number of tail_nodes: %d\n",
           head_stat->re_nodes->size, head_stat->re_edges->size);
    printf("number of pcst: %d run_time: %.6f\n",
           head_stat->num_pcst, run_time);
    free_graph_stat(head_stat);
    free(graph->costs), free(graph->prizes), free(graph->edges);
    free(graph);
}

void test_tail_proj_exact() {
    Data *graph = read_mnist_data();
    PruningMethod pruning = GWPruning;
    int g = 1, sparsity = 100, max_iter = 50, root = -1, n = graph->p;
    int m = graph->m, verbose = 0;
    double C = 2 * (sparsity - 1.) * graph->costs[0], nu = 2.5, err_tol = 1e-6;
    double epsilon = 1e-6, run_time = 0.0;
    GraphStat *tail_stat = make_graph_stat(n, m);
    tail_proj_exact(
            graph->edges, graph->costs, graph->prizes, g, C, nu,
            max_iter, err_tol, root, pruning, epsilon, n, m, verbose,
            tail_stat);
    printf("number of head_nodes: %d number of tail_nodes: %d\n",
           tail_stat->re_nodes->size, tail_stat->re_edges->size);
    printf("number of pcst: %d run_time: %.6f\n",
           tail_stat->num_pcst, run_time);
    free_graph_stat(tail_stat);
    free(graph->costs), free(graph->prizes), free(graph->edges);
    free(graph);
}

void test_tail_proj_approx() {
    Data *graph = read_mnist_data();
    PruningMethod pruning = GWPruning;
    int g = 1, sparsity = 100, max_iter = 50, root = -1, n = graph->p;
    int m = graph->m, verbose = 0;
    double C = 2 * (sparsity - 1.) * graph->costs[0], nu = 2.5, err_tol = 1e-6;
    double epsilon = 1e-6, run_time = 0.0;
    GraphStat *tail_stat = make_graph_stat(n, m);
    tail_proj_approx(
            graph->edges, graph->costs, graph->prizes, g, C, nu, max_iter,
            err_tol, root, pruning, epsilon, n, m, verbose, tail_stat);
    printf("number of head_nodes: %d number of tail_nodes: %d\n",
           tail_stat->re_nodes->size, tail_stat->re_edges->size);
    printf("number of pcst: %d run_time: %.6f\n",
           tail_stat->num_pcst, run_time);
    free_graph_stat(tail_stat);
    free(graph->costs), free(graph->prizes), free(graph->edges);
    free(graph);
}

void test_all() {
    test_head_proj_exact();
    test_head_proj_approx();
    test_tail_proj_exact();
    test_tail_proj_approx();
}

void build_grid_graph(const double *values, bool include_root, double gamma,
                      EdgePair *edges, double *prizes, double *costs,
                      int *root, int n, int height, int width) {
    *root = -1;
    if (n != (width * height)) {
        printf("number of nodes is inconsistent with width and height!");
    }
    if (include_root) {
        n += 1;
    }
    int edge_index = 0;
    for (int yy = 0; yy < height; ++yy) {
        for (int xx = 0; xx < width; ++xx) {
            int cur_index = yy * width + xx;
            prizes[cur_index] = values[yy * width + xx];
            if (xx != width - 1) {
                int next_right = cur_index + 1;
                edges[edge_index].first = cur_index;
                edges[edge_index].second = next_right;
                costs[edge_index] = 1.0;
                edge_index++;
            }
            if (yy != height - 1) {
                int next_down = cur_index + width;
                edges[edge_index].first = cur_index;
                edges[edge_index].second = next_down;
                costs[edge_index] = 1.0;
                edge_index++;
            }
        }
    }
    if (include_root) {
        *root = n - 1;
        prizes[*root] = 0.0; // TODO this may have a problem.
        double root_edge_cost = 1.0 + gamma;
        for (int ii = 0; ii < *root; ++ii) {
            edges[edge_index].first = *root;
            edges[edge_index].second = ii;
            costs[edge_index] = root_edge_cost;
            edge_index++;
        }
    }
}

bool test_cluster_grid_pcst() {
    double values[25] = {1.0, 1.0, 0.0, 0.0, 0.0,
                         1.0, 1.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 1.0, 1.0};
    bool include_root = false;
    int target_num_clusters = 2;
    double lambda = 0.5;
    double gamma = 1.0;
    PruningMethod pruning = GWPruning;
    int verbose = 0;
    int height = 5;
    int width = 5;
    int n = 25;
    int m = height * (width - 1) * 2;
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    double *prizes = malloc(sizeof(double) * n);
    double *costs = malloc(sizeof(double) * m);
    int root;
    build_grid_graph(values, include_root, gamma, edges,
                     prizes, costs, &root, n, height, width);
    GraphStat *stat = make_graph_stat(n, m);
    cluster_grid_pcst(edges, costs, prizes, n, m, target_num_clusters, lambda,
                      root, pruning, verbose, stat);
    printf("result nodes: ");
    for (int i = 0; i < stat->re_nodes->size; i++) {
        printf("%2d ", stat->re_nodes->array[i]);
    }
    printf("\nresult edges: ");
    for (int i = 0; i < stat->re_edges->size; i++) {
        printf("%2d ", stat->re_edges->array[i]);
    }
    printf("\n-----\n");
    free_graph_stat(stat);
    free(edges);
    free(prizes);
    free(costs);
    return true;
}


bool test_cluster_grid_pcst_binsearch() {
    double values[25] = {1.0, 1.0, 0.0, 0.0, 0.0,
                         1.0, 1.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 1.0, 1.0};
    int target_num_clusters = 2;
    int verbose = 2;
    int height = 5;
    int width = 5;
    int n = width * height;
    int m = height * (width - 1) * 2;
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    double *prizes = malloc(sizeof(double) * n);
    double *costs = malloc(sizeof(double) * m);
    bool include_root = false;
    double gamma = 1.0;
    int root;

    build_grid_graph(values, include_root, gamma, edges,
                     prizes, costs, &root, n, height, width);
    int k_low = 6;
    int k_high = 8;
    int max_num_iter = 10;
    GraphStat *stat = make_graph_stat(n, m);
    cluster_grid_pcst_binsearch(
            edges, costs, prizes, n, m, target_num_clusters, root, k_low,
            k_high, max_num_iter, GWPruning, verbose, stat);
    printf("result nodes: ");
    for (int i = 0; i < stat->re_nodes->size; i++) {
        printf("%2d ", stat->re_nodes->array[i]);
    }
    printf("\nresult edges: ");
    for (int i = 0; i < stat->re_edges->size; i++) {
        printf("%2d ", stat->re_edges->array[i]);
    }
    printf("\n-----\n");
    free_graph_stat(stat);
    free(edges);
    free(prizes);
    free(costs);
    return 0;
}

bool test_head_tail_binsearch() {
    double values[25] = {1.0, 1.0, 0.0, 0.0, 0.0,
                         1.0, 1.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 1.0, 1.0};
    int target_num_clusters = 2;
    int verbose = 2;
    int height = 5;
    int width = 5;
    int n = width * height;
    int m = height * (width - 1) * 2;
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    double *prizes = malloc(sizeof(double) * n);
    double *costs = malloc(sizeof(double) * m);
    bool include_root = false;
    double gamma = 1.0;
    int root;

    build_grid_graph(values, include_root, gamma, edges,
                     prizes, costs, &root, n, height, width);
    int k_low = 6;
    int k_high = 8;
    int max_num_iter = 10;
    GraphStat *stat = make_graph_stat(n, m);
    head_tail_binsearch(
            edges, costs, prizes, n, m, target_num_clusters, root, k_low,
            k_high, max_num_iter, GWPruning, verbose, stat);
    printf("result nodes: ");
    for (int i = 0; i < stat->re_nodes->size; i++) {
        printf("%2d ", stat->re_nodes->array[i]);
    }
    printf("\nresult edges: ");
    for (int i = 0; i < stat->re_edges->size; i++) {
        printf("%2d ", stat->re_edges->array[i]);
    }
    printf("\n-----\n");
    free_graph_stat(stat);
    free(edges);
    free(prizes);
    free(costs);
    return 0;
}

int main() {
    test_all();
}