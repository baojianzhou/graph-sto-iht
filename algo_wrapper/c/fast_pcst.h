/*****************************************************************************
 * Created by -- on 6/3/18.
 * This code exactly follows Ludwig Schmidt's pcst-fast algorithm:
 *      https://github.com/fraenkel-lab/pcst_fast
 *      https://github.com/ludwigschmidt/pcst-fast
 * The pcst-fast algorithm is proposed in the following paper:
 * [1]  A Fast, Adaptive Variant of the Goemans-Williamson Scheme for
 *      the Prize-Collecting Steiner Tree Problem. Chinmay Hegde, Piotr Indyk,
 *      Ludwig Schmidt Workshop of the 11th DIMACS Implementation Challenge in
 *      Collaboration with ICERM: Steiner Tree Problems, 2014
 * Licences: No Licence! The author has no any responsibility for this code.
 * this priority queue is used the following code:
 *      https://github.com/vy/libpqueue.git
*****************************************************************************/
#ifndef FAST_PCST_H
#define FAST_PCST_H

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>


typedef enum PruningMethod {
    NoPruning,
    SimplePruning,
    GWPruning,
    StrongPruning
} PruningMethod;

typedef struct {
    int *array;
    int size;
} Array;

typedef struct {
    int first;
    int second;
} EdgePair;

typedef struct {
    int first;
    double second;
} KeyPair;

typedef struct {
    KeyPair *array;
    int size;
} KeyPairArray;


typedef struct PHNode {
    struct PHNode *sibling;
    struct PHNode *child;
    struct PHNode *left_up;
    double val;
    //child_offset
    double c_offset;
    // payload
    int pay;
} PHNode;

typedef struct {
    int num_nodes;
    struct PHNode *root;
    struct PHNode **buffer;
} PairingHeap;


typedef struct {
    double next_event_val;
    bool deleted;
    PHNode *heap_node;
} EdgePart;

typedef struct {
    // active_cluster_index
    int act_c_ind;
    // inactive_cluster_index
    int inact_c_ind;
    // active_cluster_node
    int act_c_node;
    // inactive_cluster_node
    int inact_c_node;
} InactiveMergeEvent;

typedef struct {
    double val;
    int index;
    int posi;
} PQNode;

typedef struct {
    // number of elements in the queue
    int size;
    // pointer to minimal node of this queue
    PQNode **queue;
    // index all elements by its index
    PQNode **index_iter;
} PriorityQueue;

typedef struct {
    // edge_parts
    PairingHeap *e_parts;
    // active
    bool active;
    // active_start_time
    double act_s_t;
    //active_end_time
    double act_e_t;
    int merged_into;
    double prize_sum;
    // sub_cluster_moat_sum
    double sub_c_moat_sum;
    double moat;
    bool contains_root;
    int skip_up;
    double skip_up_sum;
    int merged_along;
    // child_cluster_1
    int child_c_1;
    // child_cluster_2
    int child_c_2;
    bool necessary;
} Cluster;

typedef struct {
    // root (-1 for non-root case)
    int root;
    // number of active clusters in pcst
    int g;
    // total n nodes in graph
    int n;
    // total m undirected edges in graph
    int m;
    // array of edges
    EdgePair *edges;
    // array of prizes
    double *prizes;
    // array of costs
    double *costs;
    PruningMethod pruning;
    int verbose;
    // at most 2*n number of clusters.
    Cluster *clusters;
    int clusters_size;
    // clusters_deactivation at most 3*n elements.
    PriorityQueue *c_deact;
    //clusters_next_edge_event at most 3*n elements.
    PriorityQueue *c_event;
    // current time
    double cur_time;
    // epsilon to control the precision of pcst.
    double eps;
    bool *node_good;
    bool *node_deleted;
    KeyPairArray *p3_nei;
    // path_compression_visited
    KeyPair *p_comp_visited;
    // cluster_queue
    int *c_queue;
    int *final_comp_label;
    // final_components
    Array *final_comp;
    int root_comp_index;
    //strong_pruning_parent
    KeyPair *strong_parent;
    //strong_pruning_payoff
    double *strong_pay;
    //edge_parts
    EdgePart *e_parts;
    // TODO can be improved by using stack to recycle unused nodes.
    PQNode *pq_nodes;
    PHNode *ph_nodes;
    PairingHeap *ph_heaps;
    //inactive_merge_event
    InactiveMergeEvent *inact_merge_e;
    int *edge_info;
    //inactive_merge_event_size
    int inact_m_e_len;
    // phase1_results
    int *p1_re;
    int p1_re_size;
    // phase2_results
    int *p2_re;
    int p2_re_size;
} PCST;

/**
 * To make run pcst multiple times more efficient. Graph must be connected.
 * @param edges: list of edges. node starts from 0 to n - 1
 * @param prizes: list of prizes. non-negative.
 * @param costs: list of costs on edges. must be positive.
 * @param root: -1 default non-root.
 * @param g: number of active cluster.
 * @param eps: precision, default is 1e-6.
 * @param pruning: pruning strategy. we have 4 strategies.
 * @param n: number of nodes in the graph.
 * @param m: number of edges in the graph.
 * @return constructed pcst instance.
 */
PCST *make_pcst(const EdgePair *edges, const double *prizes,
                const double *costs, int root, int g, double eps,
                PruningMethod pruning, int n, int m, int verbose);

bool run_pcst(PCST *pcst, Array *result_nodes, Array *result_edges);

void get_sum_on_edge_part(PCST *pcst, int edge_part_index, double *total_sum,
                          double *finished_moat_sum, int *cur_c_ind);

void mark_nodes_as_good(PCST *pcst, int start_c_index);

void mark_clusters_as_necessary(PCST *pcst, int start_c_index);

void mark_nodes_as_deleted(PCST *pcst, int start_n_ind, int parent_n_ind);

void label_final_comp(PCST *pcst, int start_n_ind, int new_comp_ind);

void strong_pruning_from(PCST *pcst, int start_n_ind, bool mark_as_deleted);

int find_best_comp_root(PCST *pcst, int comp_index);

bool free_pcst(PCST *pcst);

#endif //FAST_PCST_H