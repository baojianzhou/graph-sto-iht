#include "fast_pcst.h"

#define left(i)   ((unsigned int)(i) << 1u)
#define parent(i) ((unsigned int)(i) >> 1u)
#define right(i) ((unsigned int)(i) >> 1u)

// TODO Warning about the size of the buffer: 1. You need to change it if
// your graph has huge number of nodes, say 10 millions.
#define MAX_BUFFER_SIZE 100000
PHNode *buffer[MAX_BUFFER_SIZE];

// priority queue of bubble up
static inline void pq_bubble_up(PriorityQueue *q, int i) {
    int parent_node;
    PQNode *moving_node = q->queue[i];
    for (parent_node = parent(i);
         ((i > 1) && ((q->queue[parent_node]->val > moving_node->val) ||
                      (q->queue[parent_node]->val == moving_node->val &&
                       q->queue[parent_node]->index > moving_node->index)));
         i = parent_node, parent_node = parent(i)) {
        q->queue[i] = q->queue[parent_node];
        q->queue[i]->posi = i;
    }
    q->queue[i] = moving_node;
    moving_node->posi = i;
}

// priority queue of maximal child
static inline int pq_max_child(PriorityQueue *pq, int i) {
    int child_node = left(i);
    if (child_node >= pq->size) {
        return 0;
    }
    if ((child_node + 1) < pq->size &&
        ((pq->queue[child_node]->val > pq->queue[child_node + 1]->val) ||
         (pq->queue[child_node]->val == pq->queue[child_node + 1]->val &&
          pq->queue[child_node]->index > pq->queue[child_node + 1]->index))) {
        child_node++;
    }
    return child_node;
}

// priority queue of percolate down
static inline void pq_percolate_down(PriorityQueue *pq, int i) {
    int child_node;
    PQNode *moving_node = pq->queue[i];
    while ((child_node = pq_max_child(pq, i)) &&
           ((moving_node->val > pq->queue[child_node]->val) ||
            (moving_node->val == pq->queue[child_node]->val &&
             moving_node->index > pq->queue[child_node]->index))) {
        pq->queue[i] = pq->queue[child_node];
        pq->queue[i]->posi = i;
        i = child_node;
    }
    pq->queue[i] = moving_node;
    moving_node->posi = i;
}

// to make a priority queue
PriorityQueue *pq_make(int n) {
    PriorityQueue *pq = malloc(sizeof(PriorityQueue));
    pq->queue = malloc((n + 1) * sizeof(PQNode *));
    pq->index_iter = malloc((n) * sizeof(PQNode *));
    pq->size = 1;
    return pq;
}

// delete an element in priority queue
static inline void pq_delete_element(PriorityQueue *pq, int index) {
    PQNode *cur_node = pq->index_iter[index];
    int posi = cur_node->posi;
    pq->queue[posi] = pq->queue[--pq->size];
    if ((cur_node->val > pq->queue[posi]->val) ||
        (cur_node->val == pq->queue[posi]->val &&
         cur_node->index > pq->queue[posi]->index)) {
        pq_bubble_up(pq, posi);
    } else {
        pq_percolate_down(pq, posi);
    }
}

// linking of a pairing heap
static PHNode *ph_linking(PHNode *node1, PHNode *node2) {
    if (node1 == NULL) {
        return node2;
    }
    if (node2 == NULL) {
        return node1;
    }
    PHNode *smaller_node = node2, *larger_node = node1;
    if (node1->val < node2->val) {
        smaller_node = node1;
        larger_node = node2;
    }
    larger_node->sibling = smaller_node->child;
    if (larger_node->sibling != NULL) {
        larger_node->sibling->left_up = larger_node;
    }
    larger_node->left_up = smaller_node;
    smaller_node->child = larger_node;
    larger_node->val -= smaller_node->c_offset;
    larger_node->c_offset -= smaller_node->c_offset;
    return smaller_node;
}

// Warning: When you do parallel, a lock needs to be added.
static bool ph_delete_min(PairingHeap *heap, double *value, int *payload) {
    if (heap->root == NULL) {
        return false;
    }
    int buffer_size = 0, merged_children = 0;
    PHNode *cur_child = heap->root->child;
    PHNode *next_child = NULL, *result = heap->root;
    while (cur_child != NULL) {
        heap->buffer[buffer_size++] = cur_child;
        next_child = cur_child->sibling;
        cur_child->left_up = NULL;
        cur_child->sibling = NULL;
        cur_child->val += result->c_offset;
        cur_child->c_offset += result->c_offset;
        cur_child = next_child;
    }
    while (merged_children + 2 <= buffer_size) {
        heap->buffer[right(merged_children)] = ph_linking(
                heap->buffer[merged_children],
                heap->buffer[merged_children + 1]);
        merged_children += 2;
    }
    if (merged_children != buffer_size) {
        heap->buffer[right(merged_children)] = heap->buffer[merged_children];
        buffer_size = right(merged_children) + 1;
    } else {
        buffer_size = right(merged_children);
    }
    if (buffer_size > 0) {
        heap->root = heap->buffer[buffer_size - 1];
        for (int ii = buffer_size - 2; ii >= 0; --ii) {
            heap->root = ph_linking(heap->root, heap->buffer[ii]);
        }
    } else {
        heap->root = NULL;
    }
    *value = result->val;
    *payload = result->pay;
    return true;
}

//pairing heap decrease key value
static PHNode *ph_decrease_key(PairingHeap *hp,
                               PHNode *node,
                               double from_value,
                               double to_value) {
    double additional_offset = from_value - node->val;
    node->c_offset += additional_offset;
    node->val = to_value;
    if (node->left_up != NULL) {
        if (node->left_up->child == node) {
            node->left_up->child = node->sibling;
        } else {
            node->left_up->sibling = node->sibling;
        }
        if (node->sibling != NULL) {
            node->sibling->left_up = node->left_up;
        }
        node->left_up = NULL;
        node->sibling = NULL;
        hp->root = ph_linking(hp->root, node);
    }
    return node;
}

// to create a pcst instance by using malloc and calloc
PCST *make_pcst(const EdgePair *edges,
                const double *prizes,
                const double *costs,
                int root,
                int g,
                double eps,
                PruningMethod pruning,
                int n,
                int m, int verbose) {
    PCST *pcst = (PCST *) malloc(sizeof(PCST));
    pcst->root = root;
    pcst->verbose = verbose;
    pcst->g = g;
    pcst->n = n;
    pcst->m = m;
    pcst->eps = eps;
    pcst->pruning = pruning;
    pcst->clusters_size = 0;
    pcst->cur_time = 0.0;
    pcst->root_comp_index = -1;
    pcst->p1_re_size = 0;
    pcst->p2_re_size = 0;
    pcst->inact_m_e_len = 0;
    // try to copy all of the input data
    pcst->edges = malloc(sizeof(EdgePair) * m);
    pcst->costs = malloc(sizeof(double) * m);
    pcst->prizes = malloc(sizeof(double) * n);
    for (int ii = 0; ii < m; ii++) {
        pcst->edges[ii].first = edges[ii].first;
        pcst->edges[ii].second = edges[ii].second;
        pcst->costs[ii] = costs[ii];
    }
    for (int ii = 0; ii < n; ii++) {
        pcst->prizes[ii] = prizes[ii];
    }
    // number of clusters is at most 2*n
    pcst->clusters = malloc(sizeof(Cluster) * left(n));
    pcst->c_deact = pq_make(3 * n);
    pcst->c_event = pq_make(3 * n);
    pcst->node_good = malloc(sizeof(bool) * n);
    pcst->node_deleted = malloc(sizeof(bool) * n);
    pcst->p3_nei = malloc(sizeof(KeyPairArray) * n);
    pcst->p_comp_visited = malloc(sizeof(KeyPair) * n);
    // number of cluster queue is at most 2*n
    pcst->c_queue = malloc(sizeof(int) * left(n));
    pcst->final_comp_label = malloc(sizeof(int) * n);
    pcst->final_comp = malloc(sizeof(Array) * g);
    pcst->strong_parent = malloc(sizeof(KeyPair) * n);
    pcst->strong_pay = malloc(sizeof(double) * n);
    pcst->e_parts = malloc(sizeof(EdgePart) * left(m));
    pcst->edge_info = malloc(sizeof(int) * m);
    // TODO improve this part by using a stack to recycle unused nodes
    pcst->ph_nodes = malloc(sizeof(PHNode) * (10 * m));
    pcst->pq_nodes = malloc(sizeof(PQNode) * (10 * m));
    pcst->ph_heaps = malloc(sizeof(PairingHeap) * left(n));
    pcst->inact_merge_e = malloc(sizeof(InactiveMergeEvent) * left(n));
    pcst->p1_re = malloc(sizeof(int) * n);
    pcst->p2_re = malloc(sizeof(int) * n);
    pcst->verbose = verbose;
    if (pcst->verbose > 0) {
        for (int ii = 0; ii < 79; ii++) { printf("%s", "-"); } // line
        printf("\n");
        int left = (79 - 20) / 2, right = (79 - 20) / 2;
        for (int i = 0; i < left; i++) {
            printf(" ");
        }
        printf("%s", " PCST configuration ");
        for (int i = 0; i < right; i++) { printf(" "); }
        printf("\n");
        for (int ii = 0; ii < 79; ii++) { printf("%s", "-"); } // line
        printf("\n");
        printf("%-60s%-29s\n", "parameter", "val");
        for (int ii = 0; ii < 79; ii++) { printf("%s", "_"); } // line
        printf("\n");
        printf("%-60s %-6d\n", "number of nodes in graph", n);
        printf("%-60s %-6d\n", "number of edges in graph", m);
        printf("%-60s %-6d\n", "root(-1 is non-root)", root);
        printf("%-60s %-6.2e\n", "pcst epsilon", eps);
        printf("%-60s %-6d\n", "number of connected components", g);
        printf("%-60s %-6u\n", "selected pruning strategy", pcst->pruning);
        for (int ii = 0; ii < 79; ii++) { printf("%s", "_"); } // line
        printf("\n");
        printf("\n\n\n");
    }
    return pcst;
}

// to run pcst algorithm
bool run_pcst(PCST *pcst,
              Array *result_nodes,
              Array *result_edges) {
    //TODO to add a special case where prizes are all zeros.
    pcst->clusters_size = 0;
    pcst->cur_time = 0.0;
    pcst->root_comp_index = -1;
    pcst->p1_re_size = 0;
    pcst->p2_re_size = 0;
    pcst->inact_m_e_len = 0;
    for (int ii = 0; ii < pcst->n; ii++) {
        pcst->node_good[ii] = false;
        pcst->node_deleted[ii] = false;
    }
    PHNode *ph_new_node;
    PQNode *pq_new_node;
    PairingHeap *ph_new_heap;
    int ph_nodes_size = 0;
    int pq_nodes_size = 0;
    int ph_heaps_size = 0;
    for (int ii = 0; ii < 2 * pcst->n; ii++) {
        pcst->ph_heaps[ii].num_nodes = 0;
        pcst->ph_heaps[ii].root = NULL;
        pcst->ph_heaps[ii].buffer = buffer;
    }
    // mark inactive_merge_event
    for (int ii = 0; ii < pcst->m; ii++) {
        pcst->edge_info[ii] = -1;
    }
    pcst->clusters_size = pcst->n;
    for (int ii = 0; ii < pcst->n; ii++) {
        if (pcst->prizes[ii] < 0.0) {
            printf("Error: prizes should always be non-negative.\n");
            exit(EXIT_FAILURE);
        }
        pcst->clusters[ii].active = (ii != pcst->root);
        pcst->clusters[ii].act_s_t = 0.0;
        pcst->clusters[ii].act_e_t = -1.0;
        if (ii == pcst->root) { pcst->clusters[ii].act_e_t = 0.0; }
        pcst->clusters[ii].merged_into = -1;
        pcst->clusters[ii].prize_sum = pcst->prizes[ii];
        pcst->clusters[ii].sub_c_moat_sum = 0.0;
        pcst->clusters[ii].moat = 0.0;
        pcst->clusters[ii].contains_root = (ii == pcst->root);
        pcst->clusters[ii].skip_up = -1;
        pcst->clusters[ii].skip_up_sum = 0.0;
        pcst->clusters[ii].merged_along = -1;
        pcst->clusters[ii].child_c_1 = -1;
        pcst->clusters[ii].child_c_2 = -1;
        pcst->clusters[ii].necessary = false;
        // create empty pairing heap.
        ph_new_heap = &pcst->ph_heaps[ph_heaps_size++];
        pcst->clusters[ii].e_parts = ph_new_heap;
        if (pcst->clusters[ii].active) { //pq_insert
            pq_new_node = &pcst->pq_nodes[pq_nodes_size];
            pq_new_node->val = pcst->prizes[ii];
            pq_new_node->index = ii;
            pcst->c_deact->index_iter[ii] = pq_new_node;
            pcst->c_deact->queue[pcst->c_deact->size] = pq_new_node;
            pq_bubble_up(pcst->c_deact, pcst->c_deact->size++);
            pq_nodes_size++;
        }
    }
    for (int ii = 0; ii < pcst->m; ii++) {
        int uu = pcst->edges[ii].first;
        int vv = pcst->edges[ii].second;
        double cost = pcst->costs[ii];
        if (uu < 0 || vv < 0 || uu >= pcst->n
            || vv >= pcst->n || cost < 0.0) {
            char *error_1 = "node index should be [0,n-1].\n";
            char *error_2 = "edge endpoint is out of range: too large.\n";
            char *error_3 = "edge endpoint is negative.\n";
            fprintf(stderr, "%s%s%s", error_1, error_2, error_3);
            exit(EXIT_FAILURE);
        }
        EdgePart *uu_part, *vv_part;
        uu_part = pcst->e_parts + (2 * ii);
        vv_part = pcst->e_parts + (2 * ii + 1);
        Cluster *uu_cluster, *vv_cluster;
        uu_cluster = pcst->clusters + uu;
        vv_cluster = pcst->clusters + vv;
        uu_part->deleted = false, vv_part->deleted = false;
        if (uu_cluster->active && vv_cluster->active) {
            double event_time = cost / 2.0;
            uu_part->next_event_val = event_time;
            vv_part->next_event_val = event_time;
        } else if (uu_cluster->active) {
            uu_part->next_event_val = cost;
            vv_part->next_event_val = 0.0;
        } else if (vv_cluster->active) {
            uu_part->next_event_val = 0.0;
            vv_part->next_event_val = cost;
        } else {
            uu_part->next_event_val = 0.0;
            vv_part->next_event_val = 0.0;
        }
        // cur_time = 0, so the next event time
        // for each edge is the same as the next_event_val
        // ph_insert
        ph_new_node = &pcst->ph_nodes[ph_nodes_size];
        ph_new_node->sibling = NULL;
        ph_new_node->child = NULL;
        ph_new_node->left_up = NULL;
        ph_new_node->val = uu_part->next_event_val;
        ph_new_node->pay = 2 * ii;
        ph_new_node->c_offset = 0;
        PHNode *tmp = ph_linking(uu_cluster->e_parts->root, ph_new_node);
        uu_cluster->e_parts->root = tmp;
        uu_part->heap_node = ph_new_node;
        ph_nodes_size++;
        // ph_insert
        ph_new_node = &pcst->ph_nodes[ph_nodes_size];
        ph_new_node->sibling = NULL;
        ph_new_node->child = NULL;
        ph_new_node->left_up = NULL;
        ph_new_node->val = vv_part->next_event_val;
        ph_new_node->pay = 2 * ii + 1;
        ph_new_node->c_offset = 0;
        tmp = ph_linking(vv_cluster->e_parts->root, ph_new_node);
        vv_cluster->e_parts->root = tmp;
        vv_part->heap_node = ph_new_node;
        ph_nodes_size++;
    }
    for (int ii = 0; ii < pcst->n; ++ii) {
        if (pcst->clusters[ii].active &&
            pcst->clusters[ii].e_parts->root != NULL) {
            //pq_insert
            pq_new_node = &pcst->pq_nodes[pq_nodes_size];
            pq_new_node->val = pcst->clusters[ii].e_parts->root->val;
            pq_new_node->index = ii;
            pcst->c_event->index_iter[ii] = pq_new_node;
            pcst->c_event->queue[pcst->c_event->size] = pq_new_node;
            pq_bubble_up(pcst->c_event, pcst->c_event->size++);
            pq_nodes_size++;
        }
    }
    if (pcst->root >= 0 && pcst->g > 0) {
        printf("Error: g must be 0 in the rooted case.\n");
        return false;
    }
    int num_active_clusters = pcst->n;
    if (pcst->root >= 0) {
        num_active_clusters -= 1;
    }
    // all edge indices got from growth phase.
    while (num_active_clusters > pcst->g) {
        // get_next_edge_event
        double next_e_time;
        // next_edge_cluster_index, next_edge_part_index
        int n_e_c_ind, n_e_p_ind = -1;
        //pq_is_empty
        if (pcst->c_event->size == 1) {
            next_e_time = INFINITY;
            n_e_c_ind = -1;
            n_e_p_ind = -1;
        } else {
            //pq_get_min
            next_e_time = pcst->c_event->queue[1]->val;
            n_e_c_ind = pcst->c_event->queue[1]->index;
            if (pcst->clusters[n_e_c_ind].e_parts->root != NULL) {
                //ph_get_min
                next_e_time = pcst->clusters[n_e_c_ind].e_parts->root->val;
                n_e_p_ind = pcst->clusters[n_e_c_ind].e_parts->root->pay;
            }
        }
        double next_c_time; // get_next_cluster_event
        int next_cluster_index;
        if (pcst->c_deact->size == 1) { //pq_is_empty
            next_c_time = INFINITY;
            next_cluster_index = -1;
        } else {
            next_c_time = pcst->c_deact->queue[1]->val; //pq_get_min
            next_cluster_index = pcst->c_deact->queue[1]->index;
        }
        if (next_e_time < next_c_time) {
            pcst->cur_time = next_e_time;
            // remove next edge event
            pq_delete_element(pcst->c_event, n_e_c_ind);
            double tmp_value;
            int tmp_edge_part;
            ph_delete_min(pcst->clusters[n_e_c_ind].e_parts,
                          &tmp_value, &tmp_edge_part);
            if (pcst->clusters[n_e_c_ind].e_parts->root != NULL) {
                //ph_get_min
                tmp_value = pcst->clusters[n_e_c_ind].e_parts->root->val;
                tmp_edge_part = pcst->clusters[n_e_c_ind].e_parts->root->pay;
                //pq_insert
                pq_new_node = &pcst->pq_nodes[pq_nodes_size];
                pq_new_node->val = tmp_value;
                pq_new_node->index = n_e_c_ind;
                pcst->c_event->index_iter[n_e_c_ind] = pq_new_node;
                pcst->c_event->queue[pcst->c_event->size] = pq_new_node;
                pq_bubble_up(pcst->c_event, pcst->c_event->size++);
                pq_nodes_size++;
            }
            if (pcst->e_parts[n_e_p_ind].deleted) {
                continue;
            }
            // collect other_edge_part_index
            int other_e_p_ind;
            if (n_e_p_ind % 2 == 0) {
                other_e_p_ind = n_e_p_ind + 1;
            } else {
                other_e_p_ind = n_e_p_ind - 1;
            }
            double cur_edge_cost = pcst->costs[right(n_e_p_ind)];
            double sum_cur_edge_part, curr_finished_moat_sum;
            int cur_c_ind;
            get_sum_on_edge_part(pcst, n_e_p_ind, &sum_cur_edge_part,
                                 &curr_finished_moat_sum, &cur_c_ind);
            double sum_other_edge_part, other_finished_moat_sum;
            int other_c_ind;
            get_sum_on_edge_part(pcst, other_e_p_ind, &sum_other_edge_part,
                                 &other_finished_moat_sum, &other_c_ind);
            double remainder = cur_edge_cost - sum_cur_edge_part -
                               sum_other_edge_part;
            Cluster *cur_cluster = &pcst->clusters[cur_c_ind];
            Cluster *other_cluster = &pcst->clusters[other_c_ind];
            EdgePart *next_edge_part = &pcst->e_parts[n_e_p_ind];
            EdgePart *other_edge_part = &pcst->e_parts[other_e_p_ind];
            if (cur_c_ind == other_c_ind) {
                pcst->e_parts[other_e_p_ind].deleted = true;
                continue;
            }
            if (remainder < pcst->eps * cur_edge_cost || remainder == 0.0) {
                pcst->p1_re[pcst->p1_re_size++] = right(n_e_p_ind);
                pcst->e_parts[other_e_p_ind].deleted = true;
                // new_cluster_index
                int new_c_ind = pcst->clusters_size;
                // new_cluster
                Cluster *new_c = &pcst->clusters[new_c_ind];
                pcst->clusters_size++;
                // current_cluster_, other_cluster_
                Cluster *cur_c_ = &pcst->clusters[cur_c_ind];
                Cluster *oth_c_ = &pcst->clusters[other_c_ind];
                new_c->moat = 0.0;
                new_c->prize_sum = cur_c_->prize_sum + oth_c_->prize_sum;
                new_c->sub_c_moat_sum =
                        cur_c_->sub_c_moat_sum + oth_c_->sub_c_moat_sum;
                new_c->contains_root = cur_c_->contains_root ||
                                       oth_c_->contains_root;
                new_c->active = !new_c->contains_root;
                new_c->merged_along = (right(n_e_p_ind));
                new_c->child_c_1 = cur_c_ind;
                new_c->child_c_2 = other_c_ind;
                new_c->necessary = false;
                new_c->skip_up = -1;
                new_c->skip_up_sum = 0.0;
                new_c->merged_into = -1;
                cur_c_->active = false;
                cur_c_->act_e_t = pcst->cur_time + remainder;
                cur_c_->merged_into = new_c_ind;
                cur_c_->moat = cur_c_->act_e_t - cur_c_->act_s_t;
                pq_delete_element(pcst->c_deact, cur_c_ind);
                num_active_clusters -= 1;
                if (cur_c_->e_parts->root != NULL) {
                    pq_delete_element(pcst->c_event, cur_c_ind);
                }
                if (oth_c_->active) {
                    oth_c_->active = false;
                    oth_c_->act_e_t = pcst->cur_time + remainder;
                    oth_c_->moat = oth_c_->act_e_t - oth_c_->act_s_t;
                    pq_delete_element(pcst->c_deact, other_c_ind);
                    if (oth_c_->e_parts->root != NULL) {
                        pq_delete_element(pcst->c_event, other_c_ind);
                    }
                    num_active_clusters -= 1;
                } else {
                    if (!oth_c_->contains_root) {
                        // edge_event_update_time
                        double e_e_update_t = pcst->cur_time + remainder -
                                              oth_c_->act_e_t;
                        //ph_add_to_heap
                        if (oth_c_->e_parts->root != NULL) {
                            oth_c_->e_parts->root->val += e_e_update_t;
                            oth_c_->e_parts->root->c_offset += e_e_update_t;
                        }
                        pcst->inact_merge_e[pcst->inact_m_e_len].act_c_ind
                                = cur_c_ind;
                        pcst->inact_merge_e[pcst->inact_m_e_len].inact_c_ind
                                = other_c_ind;
                        int active_node_part = pcst->edges[right(
                                n_e_p_ind)].first;
                        int inactive_node_part = pcst->edges[right(
                                n_e_p_ind)].second;
                        if (n_e_p_ind % 2 == 1) {
                            int tmp = active_node_part;
                            active_node_part = inactive_node_part;
                            inactive_node_part = tmp;
                        }
                        pcst->inact_merge_e[pcst->inact_m_e_len].act_c_node
                                = active_node_part;
                        pcst->inact_merge_e[pcst->inact_m_e_len].inact_c_node
                                = inactive_node_part;
                        pcst->edge_info[right(n_e_p_ind)] =
                                pcst->inact_m_e_len;
                        pcst->inact_m_e_len++;
                    }
                }
                oth_c_->merged_into = new_c_ind;
                new_c->e_parts = cur_c_->e_parts;
                PHNode *tmp;
                tmp = ph_linking(cur_c_->e_parts->root, oth_c_->e_parts->root);
                new_c->e_parts->root = tmp;
                oth_c_->e_parts->root = NULL;
                new_c->sub_c_moat_sum += cur_c_->moat;
                new_c->sub_c_moat_sum += oth_c_->moat;
                if (new_c->active) {
                    new_c->act_s_t = pcst->cur_time + remainder;
                    double becoming_inactive_time =
                            pcst->cur_time + remainder +
                            new_c->prize_sum - new_c->sub_c_moat_sum;
                    pq_new_node = &pcst->pq_nodes[pq_nodes_size]; //pq_insert
                    pq_new_node->val = becoming_inactive_time;
                    pq_new_node->index = new_c_ind;
                    pcst->c_deact->index_iter[new_c_ind] = pq_new_node;
                    pcst->c_deact->queue[pcst->c_deact->size] = pq_new_node;
                    pq_bubble_up(pcst->c_deact, pcst->c_deact->size++);
                    pq_nodes_size++;
                    if (new_c->e_parts->root != NULL) {
                        // ph_get_min
                        double tmp_val = new_c->e_parts->root->val;
                        //pq_insert
                        pq_new_node = &pcst->pq_nodes[pq_nodes_size];
                        pq_new_node->val = tmp_val;
                        pq_new_node->index = new_c_ind;
                        pcst->c_event->index_iter[new_c_ind] = pq_new_node;
                        pcst->c_event->queue[pcst->c_event->size]
                                = pq_new_node;
                        pq_bubble_up(pcst->c_event, pcst->c_event->size++);
                        pq_nodes_size++;
                    }
                    num_active_clusters += 1;
                }
            } else if (other_cluster->active) {
                double next_event_time = pcst->cur_time + remainder / 2.0;
                next_edge_part->next_event_val =
                        sum_cur_edge_part + remainder / 2.0;
                if (cur_cluster->e_parts->root != NULL) {
                    pq_delete_element(pcst->c_event, cur_c_ind);
                }
                ph_new_node = &pcst->ph_nodes[ph_nodes_size]; //ph_insert
                ph_new_node->sibling = NULL;
                ph_new_node->child = NULL;
                ph_new_node->left_up = NULL;
                ph_new_node->val = next_event_time;
                ph_new_node->pay = n_e_p_ind;
                ph_new_node->c_offset = 0;
                PHNode *tmp;
                tmp = ph_linking(cur_cluster->e_parts->root, ph_new_node);
                cur_cluster->e_parts->root = tmp;
                next_edge_part->heap_node = ph_new_node;
                ph_nodes_size++;
                double tmp_val = -1.0;
                if (cur_cluster->e_parts->root != NULL) { //ph_get_min
                    tmp_val = cur_cluster->e_parts->root->val;
                }
                pq_new_node = &pcst->pq_nodes[pq_nodes_size]; //pq_insert
                pq_new_node->val = tmp_val;
                pq_new_node->index = cur_c_ind;
                pcst->c_event->index_iter[cur_c_ind] = pq_new_node;
                pcst->c_event->queue[pcst->c_event->size] = pq_new_node;
                pq_bubble_up(pcst->c_event, pcst->c_event->size++);
                pq_nodes_size++;
                pq_delete_element(pcst->c_event, other_c_ind);
                ph_decrease_key(other_cluster->e_parts,
                                other_edge_part->heap_node,
                                other_cluster->act_s_t +
                                other_edge_part->next_event_val
                                - other_finished_moat_sum,
                                next_event_time);
                if (other_cluster->e_parts->root != NULL) { //ph_get_min
                    tmp_val = other_cluster->e_parts->root->val;
                }
                pq_new_node = &pcst->pq_nodes[pq_nodes_size]; //pq_insert
                pq_new_node->val = tmp_val;
                pq_new_node->index = other_c_ind;
                pcst->c_event->index_iter[other_c_ind] = pq_new_node;
                pcst->c_event->queue[pcst->c_event->size] = pq_new_node;
                pq_bubble_up(pcst->c_event, pcst->c_event->size++);
                pq_nodes_size++;
                other_edge_part->next_event_val =
                        sum_other_edge_part + remainder / 2.0;
            } else {
                double next_event_time = pcst->cur_time + remainder;
                next_edge_part->next_event_val =
                        cur_edge_cost - other_finished_moat_sum;
                if (cur_cluster->e_parts->root != NULL) {
                    pq_delete_element(pcst->c_event, cur_c_ind);
                }
                ph_new_node = &pcst->ph_nodes[ph_nodes_size]; //ph_insert
                ph_new_node->sibling = NULL;
                ph_new_node->child = NULL;
                ph_new_node->left_up = NULL;
                ph_new_node->val = next_event_time;
                ph_new_node->pay = n_e_p_ind;
                ph_new_node->c_offset = 0;
                cur_cluster->e_parts->root = ph_linking(
                        cur_cluster->e_parts->root, ph_new_node);
                next_edge_part->heap_node = ph_new_node;
                ph_nodes_size++;
                double tmp_val = -1.0;
                if (cur_cluster->e_parts->root != NULL) {//ph_get_min
                    tmp_val = cur_cluster->e_parts->root->val;
                }
                pq_new_node = &pcst->pq_nodes[pq_nodes_size]; //pq_insert
                pq_new_node->val = tmp_val;
                pq_new_node->index = cur_c_ind;
                pcst->c_event->index_iter[cur_c_ind] = pq_new_node;
                pcst->c_event->queue[pcst->c_event->size] = pq_new_node;
                pq_bubble_up(pcst->c_event, pcst->c_event->size++);
                pq_nodes_size++;
                ph_decrease_key(other_cluster->e_parts,
                                other_edge_part->heap_node,
                                other_cluster->act_e_t +
                                other_edge_part->next_event_val -
                                other_finished_moat_sum,
                                other_cluster->act_e_t);
                other_edge_part->next_event_val = other_finished_moat_sum;
            }
        } else {
            pcst->cur_time = next_c_time; // cluster deactivation is first
            if (pcst->c_deact->size != 1) { // pq_delete_min
                pcst->c_deact->queue[1] =
                        pcst->c_deact->queue[--pcst->c_deact->size];
                pq_percolate_down(pcst->c_deact, 1);
            }
            Cluster *cur_c = &pcst->clusters[next_cluster_index];
            cur_c->active = false;
            cur_c->act_e_t = pcst->cur_time;
            cur_c->moat = cur_c->act_e_t - cur_c->act_s_t;
            if (cur_c->e_parts->root != NULL) {
                pq_delete_element(pcst->c_event, next_cluster_index);
            }
            num_active_clusters -= 1;
        }
    } // end of while finish to grow:
    // Finished GW clustering: final event time is pcst->cur_time
    // Mark root cluster or active clusters as good.
    result_nodes->size = 0; // get ready to extract nodes and edges.
    result_edges->size = 0;
    if (pcst->root >= 0) { // find the root cluster
        // Mark root cluster or active clusters as good.
        for (int ii = 0; ii < pcst->clusters_size; ii++) {
            if (pcst->clusters[ii].contains_root &&
                pcst->clusters[ii].merged_into == -1) {
                mark_nodes_as_good(pcst, ii);
                break;
            }
        }
    } else {
        for (int ii = 0; ii < pcst->clusters_size; ii++) {
            if (pcst->clusters[ii].active) {
                mark_nodes_as_good(pcst, ii);
            }
        }
    }
    /////////////// pruning strategy 1: NoPruning /////////////////////
    if (pcst->pruning == NoPruning) {
        bool *included = calloc((size_t) pcst->n, sizeof(bool));
        for (int ii = 0; ii < pcst->p1_re_size; ii++) {
            int uu = pcst->edges[pcst->p1_re[ii]].first;
            int vv = pcst->edges[pcst->p1_re[ii]].second;
            if (!included[uu]) {
                included[uu] = true;
                result_nodes->array[result_nodes->size++] = uu;
            }
            if (!included[vv]) {
                included[vv] = true;
                result_nodes->array[result_nodes->size++] = vv;
            }
        }
        for (int ii = 0; ii < pcst->n; ii++) {
            if (pcst->node_good[ii] && !included[ii]) {
                result_nodes->array[result_nodes->size++] = ii;
            }
        }
        free(included);
        for (int ii = 0; ii < pcst->p1_re_size; ii++) {
            result_edges->array[result_edges->size++] = pcst->p1_re[ii];
        }
        return true;
    }
    /////////////// pruning strategy 2: SimplePruning /////////////////////
    for (int ii = 0; ii < pcst->p1_re_size; ii++) {
        int uu = pcst->edges[pcst->p1_re[ii]].first;
        int vv = pcst->edges[pcst->p1_re[ii]].second;
        if (pcst->node_good[uu] && pcst->node_good[vv]) {
            pcst->p2_re[pcst->p2_re_size++] = pcst->p1_re[ii];
        }
    }
    if (pcst->pruning == SimplePruning) {
        for (int ii = 0; ii < pcst->n; ++ii) {
            if (pcst->node_good[ii]) {
                result_nodes->array[result_nodes->size++] = ii;
            }
        }
        for (int ii = 0; ii < pcst->p2_re_size; ii++) {
            result_edges->array[result_edges->size++] = pcst->p2_re[ii];
        }
        return true;
    }
    /////////////// pruning strategy 3: GWPruning /////////////////////
    int p3_nei_size[pcst->n];
    for (int ii = 0; ii < pcst->n; ii++) {
        p3_nei_size[ii] = 0;
    }
    for (int ii = 0; ii < pcst->p2_re_size; ii++) {
        p3_nei_size[pcst->edges[pcst->p2_re[ii]].first]++;
        p3_nei_size[pcst->edges[pcst->p2_re[ii]].second]++;
    }
    for (int ii = 0; ii < pcst->n; ii++) {
        pcst->p3_nei[ii].size = 0;
        pcst->p3_nei[ii].array = malloc(sizeof(KeyPair) * p3_nei_size[ii]);
    }
    for (int ii = 0; ii < pcst->p2_re_size; ii++) {
        int uu = pcst->edges[pcst->p2_re[ii]].first;
        int vv = pcst->edges[pcst->p2_re[ii]].second;
        double cur_cost = pcst->costs[pcst->p2_re[ii]];
        pcst->p3_nei[uu].array[pcst->p3_nei[uu].size].first = vv;
        pcst->p3_nei[uu].array[pcst->p3_nei[uu].size++].second = cur_cost;
        pcst->p3_nei[vv].array[pcst->p3_nei[vv].size].first = uu;
        pcst->p3_nei[vv].array[pcst->p3_nei[vv].size++].second = cur_cost;
    }
    int phase3_result[pcst->n], phase3_result_size = 0;
    if (pcst->pruning == GWPruning) {
        for (int ii = pcst->p2_re_size - 1; ii >= 0; ii--) {
            int cur_e_ind = pcst->p2_re[ii];
            int uu = pcst->edges[cur_e_ind].first;
            int vv = pcst->edges[cur_e_ind].second;
            if (pcst->node_deleted[uu] && pcst->node_deleted[vv]) {
                continue;
            }
            if (pcst->edge_info[cur_e_ind] < 0) {
                mark_clusters_as_necessary(pcst, uu);
                mark_clusters_as_necessary(pcst, vv);
                phase3_result[phase3_result_size++] = cur_e_ind;
            } else {
                int tmp = pcst->edge_info[cur_e_ind];
                int active_side_node = pcst->inact_merge_e[tmp].act_c_node;
                int inact_side_node = pcst->inact_merge_e[tmp].inact_c_node;
                int inact_c_ind = pcst->inact_merge_e[tmp].inact_c_ind;
                if (pcst->clusters[inact_c_ind].necessary) {
                    phase3_result[phase3_result_size++] = cur_e_ind;
                    mark_clusters_as_necessary(pcst, inact_side_node);
                    mark_clusters_as_necessary(pcst, active_side_node);
                } else {
                    mark_nodes_as_deleted(pcst, inact_side_node,
                                          active_side_node);
                }
            }
        }
        for (int ii = 0; ii < pcst->n; ii++) {
            if (!pcst->node_deleted[ii] && pcst->node_good[ii]) {
                result_nodes->array[result_nodes->size++] = ii;
            }
        }
        for (int ii = 0; ii < phase3_result_size; ii++) {
            result_edges->array[result_edges->size++] = phase3_result[ii];
        }
        for (int ii = 0; ii < pcst->n; ii++) {
            free(pcst->p3_nei[ii].array);
        }
        return true;
    }
    /////////////// pruning strategy 4: StrongPruning /////////////////////
    if (pcst->pruning == StrongPruning) {
        for (int kk = 0; kk < pcst->n; kk++) {
            pcst->final_comp_label[kk] = -1;
            pcst->strong_parent[kk].first = -1;
            pcst->strong_parent[kk].second = -1.0;
            pcst->strong_pay[kk] = -1.0;
        }
        pcst->root_comp_index = -1;
        int final_comps_size = 0;
        for (int ii = 0; ii < pcst->g; ii++) {
            pcst->final_comp[ii].size = 0;
            pcst->final_comp[ii].array
                    = malloc(sizeof(int) * (pcst->p2_re_size + 1));
        }
        for (int ii = 0; ii < pcst->p2_re_size; ii++) {
            int cur_node_ind = pcst->edges[pcst->p2_re[ii]].first;
            if (pcst->final_comp_label[cur_node_ind] == -1) {
                final_comps_size++;
                label_final_comp(pcst, cur_node_ind, final_comps_size - 1);
            }
        }
        for (int ii = 0; ii < final_comps_size; ii++) {
            if (ii == pcst->root_comp_index) {
                strong_pruning_from(pcst, pcst->root, true);
            } else {
                int best_comp_root = find_best_comp_root(pcst, ii);
                strong_pruning_from(pcst, best_comp_root, true);
            }
        }
        for (int ii = 0; ii < pcst->p2_re_size; ++ii) {
            int cur_edge_index = pcst->p2_re[ii];
            int uu = pcst->edges[cur_edge_index].first;
            int vv = pcst->edges[cur_edge_index].second;
            if (pcst->node_deleted[uu] || pcst->node_deleted[vv]) {
            } else {
                phase3_result[phase3_result_size++] = cur_edge_index;
            }
        }
        for (int ii = 0; ii < pcst->n; ii++) {
            if (!pcst->node_deleted[ii] && pcst->node_good[ii]) {
                result_nodes->array[result_nodes->size++] = ii;
            }
        }
        for (int ii = 0; ii < phase3_result_size; ii++) {
            result_edges->array[result_edges->size++] = phase3_result[ii];
        }
        for (int ii = 0; ii < pcst->g; ii++) {
            if (pcst->final_comp[ii].array != NULL) {
                free(pcst->final_comp[ii].array);
            }
        }
        return true;
    }
    printf("Error: unknown pruning scheme.\n");
    return false;
}


void get_sum_on_edge_part(PCST *pcst,
                          int edge_part_index,
                          double *total_sum,
                          double *finished_moat_sum,
                          int *cur_c_ind) {
    int endpoint = pcst->edges[right(edge_part_index)].first, ii;
    if (edge_part_index % 2 == 1) {
        endpoint = pcst->edges[right(edge_part_index)].second;
    }
    *total_sum = 0.0;
    *cur_c_ind = endpoint;
    int path_size = 0;
    while (pcst->clusters[*cur_c_ind].merged_into != -1) {
        pcst->p_comp_visited[path_size].first = *cur_c_ind;
        pcst->p_comp_visited[path_size++].second = *total_sum;
        if (pcst->clusters[*cur_c_ind].skip_up >= 0) {
            *total_sum += pcst->clusters[*cur_c_ind].skip_up_sum;
            *cur_c_ind = pcst->clusters[*cur_c_ind].skip_up;
        } else {
            *total_sum += pcst->clusters[*cur_c_ind].moat;
            *cur_c_ind = pcst->clusters[*cur_c_ind].merged_into;
        }
    }
    for (ii = 0; ii < path_size; ii++) {
        int v_c_ind = pcst->p_comp_visited[ii].first; //visited_cluster_index
        double visited_sum = pcst->p_comp_visited[ii].second;
        pcst->clusters[v_c_ind].skip_up = *cur_c_ind;
        pcst->clusters[v_c_ind].skip_up_sum = *total_sum - visited_sum;
    }
    if (pcst->clusters[*cur_c_ind].active) {
        *finished_moat_sum = *total_sum;
        *total_sum += pcst->cur_time - pcst->clusters[*cur_c_ind].act_s_t;
    } else {
        *total_sum += pcst->clusters[*cur_c_ind].moat;
        *finished_moat_sum = *total_sum;
    }
}


void mark_nodes_as_good(PCST *pcst, int start_c_index) {
    int q_size = 0, q_index = 0;
    pcst->c_queue[q_size++] = start_c_index; // start_cluster_index
    while (q_index < q_size) {
        int cur_c_ind = pcst->c_queue[q_index++];
        if (pcst->clusters[cur_c_ind].merged_along >= 0) {
            pcst->c_queue[q_size++] = pcst->clusters[cur_c_ind].child_c_1;
            pcst->c_queue[q_size++] = pcst->clusters[cur_c_ind].child_c_2;
        } else {
            pcst->node_good[cur_c_ind] = true;
        }
    }
}

// pcst start_cluster_index
void mark_clusters_as_necessary(PCST *pcst, int start_c_index) {
    int cur_c_index = start_c_index;
    while (!pcst->clusters[cur_c_index].necessary) {
        pcst->clusters[cur_c_index].necessary = true;
        if (pcst->clusters[cur_c_index].merged_into >= 0) {
            cur_c_index = pcst->clusters[cur_c_index].merged_into;
        } else {
            return;
        }
    }
}

void mark_nodes_as_deleted(PCST *pcst,
                           int start_node_index,
                           int parent_n_ind) {
    pcst->node_deleted[start_node_index] = true;
    int ii, queue_size = 0, queue_index = 0;
    pcst->c_queue[queue_size++] = start_node_index;
    while (queue_index < queue_size) {
        int cur_n_index = pcst->c_queue[queue_index++]; // cur_node_index
        for (ii = 0; ii < pcst->p3_nei[cur_n_index].size; ii++) {
            int next_node_index = pcst->p3_nei[cur_n_index].array[ii].first;
            if (next_node_index == parent_n_ind ||
                pcst->node_deleted[next_node_index]) {
                continue;
            }
            pcst->node_deleted[next_node_index] = true;
            pcst->c_queue[queue_size++] = next_node_index;
        }
    }
}

void label_final_comp(PCST *pcst, int start_n_ind, int new_comp_ind) {
    int queue_next = 0, queue_size = 0;
    pcst->c_queue[queue_size++] = start_n_ind; // start_node_index
    pcst->final_comp_label[start_n_ind] = new_comp_ind;
    while (queue_next < queue_size) {
        int cur_node_index = pcst->c_queue[queue_next++];
        int cur_size = pcst->final_comp[new_comp_ind].size++;
        pcst->final_comp[new_comp_ind].array[cur_size] = cur_node_index;
        if (cur_node_index == pcst->root) {
            pcst->root_comp_index = new_comp_ind;
        }
        for (int ii = 0; ii < pcst->p3_nei[cur_node_index].size; ++ii) {
            int n_node_ind = pcst->p3_nei[cur_node_index].array[ii].first;
            if (pcst->final_comp_label[n_node_ind] == -1) {
                pcst->c_queue[queue_size++] = n_node_ind;
                pcst->final_comp_label[n_node_ind] = new_comp_ind;
            }
        }
    }
}

void strong_pruning_from(PCST *pcst, int start_n_ind, bool mark_as_deleted) {
    bool stack_bool[pcst->n];
    int stack_size = 0, stack_int[pcst->n];
    stack_bool[stack_size] = true;
    stack_int[stack_size++] = start_n_ind; // start_node_index
    pcst->strong_parent[start_n_ind].first = -1;
    pcst->strong_parent[start_n_ind].second = 0.0;
    while (stack_size != 0) {
        bool begin = stack_bool[stack_size - 1];
        int cur_n_ind = stack_int[stack_size - 1]; // current_node_index
        stack_size--; // pop back
        if (begin) {
            stack_bool[stack_size] = false;
            stack_int[stack_size++] = cur_n_ind;
            for (int ii = 0; ii < pcst->p3_nei[cur_n_ind].size; ++ii) {
                int n_node_ind = pcst->p3_nei[cur_n_ind].array[ii].first;
                double next_cost = pcst->p3_nei[cur_n_ind].array[ii].second;
                if (n_node_ind == pcst->strong_parent[cur_n_ind].first) {
                    continue;
                }
                pcst->strong_parent[n_node_ind].first = cur_n_ind;
                pcst->strong_parent[n_node_ind].second = next_cost;
                stack_bool[stack_size] = true;
                stack_int[stack_size++] = n_node_ind;
            }
        } else {
            pcst->strong_pay[cur_n_ind] = pcst->prizes[cur_n_ind];
            for (int ii = 0; ii < pcst->p3_nei[cur_n_ind].size; ++ii) {
                int n_node_ind = pcst->p3_nei[cur_n_ind].array[ii].first;
                double next_cost = pcst->p3_nei[cur_n_ind].array[ii].second;
                if (n_node_ind == pcst->strong_parent[cur_n_ind].first) {
                    continue;
                } //next_payoff
                double n_payoff = pcst->strong_pay[n_node_ind] - next_cost;
                if (n_payoff <= 0.0) {
                    if (mark_as_deleted) {
                        mark_nodes_as_deleted(pcst, n_node_ind, cur_n_ind);
                    }
                } else {
                    pcst->strong_pay[cur_n_ind] += n_payoff;
                }
            }
        }
    }
}

int find_best_comp_root(PCST *pcst, int comp_index) {
    int stack[pcst->n];
    int cur_best_root_ind = pcst->final_comp[comp_index].array[0];
    int s2_size = 0; // stack2_size
    strong_pruning_from(pcst, cur_best_root_ind, false);
    double cur_best_value = pcst->strong_pay[cur_best_root_ind];
    for (int ii = 0; ii < pcst->p3_nei[cur_best_root_ind].size; ii++) {
        stack[s2_size++] = pcst->p3_nei[cur_best_root_ind].array[ii].first;
    }
    while (s2_size != 0) {
        int cur_n_ind = stack[s2_size - 1]; // current_node_index
        s2_size--; // pop back
        int cur_p_ind = pcst->strong_parent[cur_n_ind].first;
        double parent_edge_cost = pcst->strong_parent[cur_n_ind].second;
        double parent_val_without_cur_node = pcst->strong_pay[cur_p_ind];
        double cur_node_net_payoff =
                pcst->strong_pay[cur_n_ind] - parent_edge_cost;
        if (cur_node_net_payoff > 0.0) {
            parent_val_without_cur_node -= cur_node_net_payoff;
        }
        if (parent_val_without_cur_node > parent_edge_cost) {
            pcst->strong_pay[cur_n_ind] +=
                    parent_val_without_cur_node - parent_edge_cost;
        }
        if (pcst->strong_pay[cur_n_ind] > cur_best_value) {
            cur_best_root_ind = cur_n_ind;
            cur_best_value = pcst->strong_pay[cur_n_ind];
        }
        for (int ii = 0; ii < pcst->p3_nei[cur_n_ind].size; ii++) {
            int next_node_index = pcst->p3_nei[cur_n_ind].array[ii].first;
            if (next_node_index != cur_p_ind) {
                stack[s2_size++] = next_node_index;
            }
        }
    }
    return cur_best_root_ind;
}

bool free_pcst(PCST *pcst) {
    // to free all of the memory used in pcst.
    free(pcst->c_event->index_iter);
    pcst->c_event->index_iter = NULL;
    free(pcst->c_event->queue);
    pcst->c_event->queue = NULL;
    free(pcst->c_event);
    pcst->c_event = NULL;
    free(pcst->c_deact->index_iter);
    pcst->c_deact->index_iter = NULL;
    free(pcst->c_deact->queue);
    pcst->c_deact->queue = NULL;
    free(pcst->c_deact);
    pcst->c_deact = NULL;
    free(pcst->p2_re);
    pcst->p2_re = NULL;
    free(pcst->p1_re);
    pcst->p1_re = NULL;
    free(pcst->inact_merge_e);
    pcst->inact_merge_e = NULL;
    free(pcst->ph_heaps);
    pcst->ph_heaps = NULL;
    free(pcst->pq_nodes);
    pcst->pq_nodes = NULL;
    free(pcst->ph_nodes);
    pcst->ph_nodes = NULL;
    free(pcst->edge_info);
    pcst->edge_info = NULL;
    free(pcst->e_parts);
    pcst->e_parts = NULL;
    free(pcst->strong_pay);
    pcst->strong_pay = NULL;
    free(pcst->strong_parent);
    pcst->strong_parent = NULL;
    free(pcst->final_comp);
    pcst->final_comp = NULL;
    free(pcst->final_comp_label);
    pcst->final_comp_label = NULL;
    free(pcst->c_queue);
    pcst->c_queue = NULL;
    free(pcst->p_comp_visited);
    pcst->p_comp_visited = NULL;
    free(pcst->p3_nei);
    pcst->p3_nei = NULL;
    free(pcst->node_deleted);
    pcst->node_deleted = NULL;
    free(pcst->node_good);
    pcst->node_good = NULL;
    free(pcst->clusters);
    pcst->clusters = NULL;
    free(pcst->prizes);
    pcst->prizes = NULL;
    free(pcst->costs);
    pcst->costs = NULL;
    free(pcst->edges);
    pcst->edges = NULL;
    free(pcst);
    return true;
}

