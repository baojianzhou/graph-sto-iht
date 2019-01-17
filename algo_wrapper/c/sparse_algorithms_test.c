//
// Created by baojian on 8/24/18.
//
#include "math_utils.h"
#include "sparse_algorithms.h"


typedef struct {
    int p;
    int m;
    int num_tr;
    EdgePair *edges;
    double *costs;
    double *w0;
    double *x_tr;
    double *y_tr;
} Data;

Data *gen_test_case(char *file_name) {
    int num_tr = -1, p = -1, m = -1, tr_ind = 0;
    char *line = NULL, tokens[1100][20];
    EdgePair *edges = NULL;
    double *costs = NULL, *x_tr = NULL, *y_tr = NULL, *w0 = NULL;
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
        if (strcmp("P", tokens[0]) == 0) {
            num_tr = (int) strtol(tokens[1], NULL, 10);
            p = (int) strtol(tokens[2], NULL, 10);
            m = (int) strtol(tokens[3], NULL, 10);
            edges = malloc(sizeof(EdgePair) * m);
            costs = malloc(sizeof(double) * m);
            x_tr = malloc(sizeof(double) * (num_tr * p));
            y_tr = malloc(sizeof(double) * num_tr);
            w0 = malloc(sizeof(double) * (p + 1));
            continue;
        }
        if (strcmp("x_tr", tokens[0]) == 0) {
            for (int j = 0; j < p; j++) {
                x_tr[tr_ind * p + j] = strtod(tokens[j + 1], NULL);
            }
            y_tr[tr_ind] = strtod(tokens[p + 1], NULL);
            tr_ind++;
            continue;
        }
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
            w0[node] = prize;
            continue;
        }
    }
    fclose(fp);
    Data *data = malloc(sizeof(Data));
    data->m = m;
    data->p = p;
    data->num_tr = num_tr;
    data->edges = edges;
    data->w0 = w0;
    data->costs = costs;
    data->x_tr = x_tr;
    data->y_tr = y_tr;
    for (int i = 0; i < num_tr; i++) {
        printf("norm_x_tr[%d]: %.4f y_tr[%d]: %.1f\n",
               i, norm_l2(x_tr + i * p, p), i, y_tr[i]);
    }
    return data;
}

int main() {
    clock_t start_time = clock();
    char *file_name = "/network/rit/lab/ceashpc/bz383376/"
                      "data/icdm18/simu/test_case.txt";
    Data *data;
    data = gen_test_case(file_name);
    int g = 1, s = 20, max_iter = 50, verbose = 0;
    double tol = 1e-6, lr = 0.1, eta = 1e-3;
    double *wt = malloc(sizeof(double) * (data->p + 1));
    double *losses = malloc(sizeof(double) * max_iter);
    double run_time_head, run_time_tail, total_time;
    Array *re_nodes = malloc(sizeof(Array));
    re_nodes->array = malloc(sizeof(int) * data->p);
    Array *re_edges = malloc(sizeof(Array));
    re_edges->array = malloc(sizeof(int) * data->p);
    algo_batch_graph_iht_logit(
            data->edges, data->costs, g, s, data->p, data->m, data->x_tr,
            data->y_tr, data->w0, data->num_tr, tol, max_iter, lr, eta,
            verbose, wt, re_nodes, losses, &run_time_head, &run_time_tail,
            &total_time);
    printf("run time: %.2f\n",
           (double) (clock() - start_time) / CLOCKS_PER_SEC);
    // free
    free(re_edges->array), free(re_nodes->array);
    free(re_edges), free(re_nodes);
    free(losses);
    free(wt);
    // free data
    free(data->edges);
    free(data->costs);
    free(data->x_tr);
    free(data->y_tr);
    free(data->w0);
    free(data);
    printf("run time: %.2f\n",
           (double) (clock() - start_time) / CLOCKS_PER_SEC);
    return 0;
}