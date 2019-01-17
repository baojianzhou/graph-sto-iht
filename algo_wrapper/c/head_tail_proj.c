#include <unistd.h>
#include "head_tail_proj.h"
#include "sort.h"

GraphStat *make_graph_stat(int p, int m) {
    GraphStat *stat = malloc(sizeof(GraphStat));
    stat->num_pcst = 0;
    stat->re_edges = malloc(sizeof(Array));
    stat->re_edges->size = 0;
    stat->re_edges->array = malloc(sizeof(int) * p);
    stat->re_nodes = malloc(sizeof(Array));
    stat->re_nodes->size = 0;
    stat->re_nodes->array = malloc(sizeof(int) * p);
    stat->run_time = 0;
    stat->costs = malloc(sizeof(double) * m);
    stat->prizes = malloc(sizeof(double) * p);
    return stat;
}

bool free_graph_stat(GraphStat *graph_stat) {
    free(graph_stat->re_nodes->array);
    free(graph_stat->re_nodes);
    free(graph_stat->re_edges->array);
    free(graph_stat->re_edges);
    free(graph_stat->costs);
    free(graph_stat->prizes);
    free(graph_stat);
    return true;
}

// compare function for descending sorting.
int _comp_descend(const void *a, const void *b) {
    if (((data_pair *) a)->val < ((data_pair *) b)->val) {
        return 1;
    } else {
        return -1;
    }
}

// get positive minimum prize in prizes vector.
double min_pi(
        const double *prizes, double *total_prizes, int n, double err_tol,
        int verbose) {
    *total_prizes = 0.0;
    double positive_min = INFINITY;
    for (int ii = 0; ii < n; ii++) {
        *total_prizes += prizes[ii];
        if ((prizes[ii] < positive_min) && (prizes[ii] > 0.0)) {
            positive_min = prizes[ii];
        }
    }
    /**
     * Warning: There is a precision issue here. We may need to define a
     * minimum precision. In our experiment,  we found that some very
     * small positive number could be like 1.54046e-310, 1.54046e-310.
     * In this case, the fast-pcst cannot stop!!!
     */
    if (positive_min < err_tol) {
        if (verbose > 0) {
            printf("warning too small positive val found.\n");
        }
        positive_min = err_tol;
    }
    return positive_min;
}

// deep first search for finding a tour.
bool dfs_tour(
        const EdgePair *edges, int n, Array *tree_nodes, Array *tree_edges,
        Array *tour_nodes, Array *tour_edges) {
    /**
     * This method is to find a euler tour given a tree. This method is
     * proposed in the following paper:
     *  Authors : Edmonds, Jack, and Ellis L. Johnson.
     *  Title : Matching, Euler tours and the Chinese postman
     *  Journal: Mathematical programming 5.1 (1973): 88-124.
     */
    //Make sure the tree has at least two nodes.
    if (tree_nodes->size <= 1) {
        printf("error: The input tree has at least two nodes.");
        exit(0);
    }
    typedef struct {
        int first;
        int second;
        bool third;
    } Tuple;
    typedef struct {
        Tuple *array;
        size_t size;
    } Nei;
    int i, *nei_size = calloc((size_t) n, sizeof(int));
    for (i = 0; i < tree_edges->size; i++) {
        nei_size[edges[tree_edges->array[i]].first]++;
        nei_size[edges[tree_edges->array[i]].second]++;
    }
    Nei *adj = malloc(sizeof(Nei) * n);
    for (i = 0; i < n; i++) {
        adj[i].size = 0;
        adj[i].array = malloc(sizeof(Tuple) * nei_size[i]);
    }
    for (i = 0; i < tree_edges->size; i++) {
        // each tuple is: (indexed node, edge_index, is_visited)
        int uu = edges[tree_edges->array[i]].first;
        int vv = edges[tree_edges->array[i]].second;
        Tuple nei_v, nei_u;
        nei_v.second = tree_edges->array[i];
        nei_u.second = tree_edges->array[i];
        nei_v.first = vv, nei_v.third = false;
        nei_u.first = uu, nei_u.third = false;
        adj[uu].array[adj[uu].size++] = nei_v;  // edge u --> v
        adj[vv].array[adj[vv].size++] = nei_u;  // edge v --> u
    }
    // The first element as tour's root.
    int start_node = tree_nodes->array[0];
    bool *visited = calloc((size_t) n, sizeof(bool));
    tour_nodes->array[tour_nodes->size++] = start_node;
    while (true) {
        bool flag_1 = false;
        visited[start_node] = true;
        // iterate the adj of each node. in this loop, we check if
        // there exists any its neighbor which has not been visited.
        for (i = 0; i < (int) adj[start_node].size; i++) {
            int next_node = adj[start_node].array[i].first;
            int edge_index = adj[start_node].array[i].second;
            if (!visited[next_node]) { // first time to visit this node.
                visited[next_node] = true; // mark it as visited.
                tour_nodes->array[tour_nodes->size++] = next_node;
                tour_edges->array[tour_edges->size++] = edge_index;
                adj[start_node].array[i].third = true; // mark it as labeled.
                start_node = next_node;
                flag_1 = true;
                break;
            }
        }
        // all neighbors are visited. Then we check if
        // there exists adj which is false nodes.
        if (!flag_1) {
            bool flag_2 = false;
            for (i = 0; i < (int) adj[start_node].size; i++) {
                int next_node = adj[start_node].array[i].first;
                int edge_index = adj[start_node].array[i].second;
                bool is_visited = adj[start_node].array[i].third;
                // there exists a neighbor. has false node
                if (!is_visited) {
                    adj[start_node].array[i].third = true;
                    tour_nodes->array[tour_nodes->size++] = next_node;
                    tour_edges->array[tour_edges->size++] = edge_index;
                    start_node = next_node;
                    flag_2 = true;
                    break;
                }
            }
            // all nodes are visited and there is no false nodes.
            if (!flag_2) {
                break;
            }
        }
    }
    free(visited);
    for (i = 0; i < n; i++) { free(adj[i].array); }
    free(adj);
    free(nei_size);
    return true;
}

// find a dense tree
bool prune_tree(
        const EdgePair *edges, const double *prizes, const double *costs,
        int n, int m, double c_prime, Array *tree_nodes, Array *tree_edges) {
    Array *tour_nodes = malloc(sizeof(Array));
    Array *tour_edges = malloc(sizeof(Array));
    tour_nodes->size = 0, tour_edges->size = 0;
    tour_nodes->array = malloc(sizeof(int) * (2 * tree_nodes->size - 1));
    tour_edges->array = malloc(sizeof(int) * (2 * tree_nodes->size - 2));
    dfs_tour(edges, n, tree_nodes, tree_edges, tour_nodes, tour_edges);
    // calculating pi_prime.
    double *pi_prime = malloc(sizeof(double) * (2 * tour_nodes->size - 1));
    int pi_prime_size = 0;
    bool *tmp_vector = calloc((size_t) n, sizeof(bool));
    for (int ii = 0; ii < tour_nodes->size; ii++) {
        // first time show in the tour.
        if (!tmp_vector[tour_nodes->array[ii]]) {
            pi_prime[pi_prime_size++] = prizes[tour_nodes->array[ii]];
            tmp_vector[tour_nodes->array[ii]] = true;
        } else {
            pi_prime[pi_prime_size++] = 0.0;
        }
    }
    double prize_t = 0.0, cost_t = 0.0;
    for (int ii = 0; ii < tree_nodes->size; ii++) {
        prize_t += prizes[tree_nodes->array[ii]];
    }
    for (int ii = 0; ii < tree_edges->size; ii++) {
        cost_t += costs[tree_edges->array[ii]];
    }
    tree_nodes->size = 0;
    tree_edges->size = 0;
    double phi = prize_t / cost_t;
    for (int ii = 0; ii < tour_nodes->size; ii++) {
        if (prizes[tour_nodes->array[ii]] >= ((c_prime * phi) / 6.)) {
            // create a single node tree.
            tree_nodes->array[tree_nodes->size++] = tour_nodes->array[ii];
            free(tmp_vector);
            free(pi_prime);
            free(tour_nodes->array);
            free(tour_edges->array);
            free(tour_nodes);
            free(tour_edges);
            return true;
        }
    }
    Array *p_l = malloc(sizeof(Array));
    p_l->size = 0;
    p_l->array = malloc(sizeof(int) * (2 * tour_nodes->size - 1));
    for (int i = 0; i < tour_nodes->size; i++) {
        p_l->array[p_l->size++] = i;
        double pi_prime_pl = 0.0;
        for (int ii = 0; ii < p_l->size; ii++) {
            pi_prime_pl += pi_prime[p_l->array[ii]];
        }
        double c_prime_pl = 0.0;
        if (p_l->size >= 2) { // <= 1: there is no edge.
            for (int j = 0; j < p_l->size - 1; j++) {
                c_prime_pl += costs[tour_edges->array[p_l->array[j]]];
            }
        }
        if (c_prime_pl > c_prime) { // start a new sublist
            p_l->size = 0;
        } else if (pi_prime_pl >= ((c_prime * phi) / 6.)) {
            bool *added_nodes = calloc((size_t) n, sizeof(bool));
            bool *added_edges = calloc((size_t) m, sizeof(bool));
            for (int j = 0; j < p_l->size; j++) {
                int cur_node = tour_nodes->array[p_l->array[j]];
                if (!added_nodes[cur_node]) {
                    added_nodes[cur_node] = true;
                    tree_nodes->array[tree_nodes->size++] = cur_node;
                }
                int cur_edge = tour_edges->array[p_l->array[j]];
                if (!added_edges[cur_edge]) {
                    added_edges[cur_edge] = true;
                    tree_edges->array[tree_edges->size++] = cur_edge;
                }
            }
            tree_edges->size--; // pop the last edge
            free(added_edges);
            free(added_nodes);
            free(tmp_vector);
            free(pi_prime);
            free(tour_nodes->array);
            free(tour_edges->array);
            free(tour_nodes);
            free(tour_edges);
            free(p_l->array);
            free(p_l);
            return true;
        }
    }
    printf("Error: Never reach at this point.\n"); //Merge procedure.
    exit(0);
}


//sort g trees in the forest
Tree *sort_forest(
        const EdgePair *edges, const double *prizes, const double *costs,
        int g, int n, Array *f_nodes, Array *f_edges, int *sorted_ind) {
    typedef struct {
        int *array;
        size_t size;
    } Nei;
    int *neighbors_size = calloc((size_t) n, sizeof(int));
    Nei *adj = malloc(sizeof(Nei) * n);
    for (int ii = 0; ii < f_edges->size; ii++) {
        neighbors_size[edges[f_edges->array[ii]].first]++;
        neighbors_size[edges[f_edges->array[ii]].second]++;
    }
    for (int ii = 0; ii < n; ii++) {
        adj[ii].size = 0;
        adj[ii].array = malloc(sizeof(int) * neighbors_size[ii]);
    }
    for (int ii = 0; ii < f_edges->size; ii++) {
        int uu = edges[f_edges->array[ii]].first;
        int vv = edges[f_edges->array[ii]].second;
        adj[uu].array[adj[uu].size++] = vv;  // edge u --> v
        adj[vv].array[adj[vv].size++] = uu;  // edge v --> u
    }
    int t = 0; // component id
    //label nodes to the components id.
    int *comp = calloc((size_t) n, sizeof(int));
    bool *visited = calloc((size_t) n, sizeof(bool));
    int *stack = malloc(sizeof(int) * n), stack_size = 0;
    int *comp_size = calloc((size_t) g, sizeof(int));
    // dfs algorithm to get cc
    for (int ii = 0; ii < f_nodes->size; ii++) {
        int cur_node = f_nodes->array[ii];
        if (!visited[cur_node]) {
            stack[stack_size++] = cur_node;
            while (stack_size != 0) { // check empty
                int s = stack[stack_size - 1];
                stack_size--;
                if (!visited[s]) {
                    visited[s] = true;
                    comp[s] = t;
                    comp_size[t]++;
                }
                for (int k = 0; k < (int) adj[s].size; k++) {
                    if (!visited[adj[s].array[k]]) {
                        stack[stack_size++] = adj[s].array[k];
                    }
                }
            }
            t++; // to label component id.
        }
    }
    Tree *trees = malloc(sizeof(Tree) * g);
    for (int ii = 0; ii < g; ii++) {
        int tree_size = comp_size[ii];
        trees[ii].nodes = malloc(sizeof(Array));
        trees[ii].edges = malloc(sizeof(Array));
        trees[ii].nodes->size = 0;
        trees[ii].edges->size = 0;
        trees[ii].nodes->array = malloc(sizeof(int) * tree_size);
        trees[ii].edges->array = malloc(sizeof(int) * (tree_size - 1));
        trees[ii].prize = 0.0;
        trees[ii].cost = 0.0;
    }
    // insert nodes into trees.
    for (int ii = 0; ii < f_nodes->size; ii++) {
        int tree_i = comp[f_nodes->array[ii]];
        int cur_node = f_nodes->array[ii];
        trees[tree_i].nodes->array[trees[tree_i].nodes->size++] = cur_node;
        trees[tree_i].prize += prizes[f_nodes->array[ii]];
    }
    // insert edges into trees.
    for (int ii = 0; ii < f_edges->size; ii++) {
        // random select one endpoint
        int uu = edges[f_edges->array[ii]].first;
        int tree_i = comp[uu], cur_edge = f_edges->array[ii];
        trees[tree_i].edges->array[trees[tree_i].edges->size++] = cur_edge;
        trees[tree_i].cost += costs[f_edges->array[ii]];
    }

    data_pair *w_pairs = (data_pair *) malloc(sizeof(data_pair) * g);
    for (int i = 0; i < g; i++) {
        if (trees[i].cost > 0.0) { // tree weight
            w_pairs[i].val = trees[i].prize / trees[i].cost;
        } else { // for a single node tree
            w_pairs[i].val = INFINITY;
        }
        w_pairs[i].val_index = i;
    }
    qsort(w_pairs, (size_t) g, sizeof(data_pair), &_comp_descend);
    for (int i = 0; i < g; i++) {
        sorted_ind[i] = w_pairs[i].val_index;
    }
    free(w_pairs);
    free(comp_size);
    free(stack);
    free(visited);
    free(comp);
    for (int ii = 0; ii < n; ii++) { free(adj[ii].array); }
    free(adj);
    free(neighbors_size);
    return trees;
}

bool prune_forest(
        const EdgePair *edges, const double *prizes, const double *costs,
        int g, int n, int m, double C, Array *f_nodes, Array *f_edges) {
    // case 1: usually, there is only one tree. then forest is a tree.
    int i, j;
    double cost_f = 0.0, prize_f = 0.0;
    for (i = 0; i < f_nodes->size; i++) {
        prize_f += prizes[f_nodes->array[i]];
    }
    for (i = 0; i < f_edges->size; i++) {
        cost_f += costs[f_edges->array[i]];
    }
    if (g == 1) {
        // single node forest or it is already good enough
        if (cost_f <= C) {
            return true;
        } else if (0.0 < C) {
            // must have at least two nodes
            prune_tree(edges, prizes, costs, n, m, C, f_nodes, f_edges);
            return true;
        } else {
            //return a maximal node
            int max_node = f_nodes->array[0];
            double max_prize = prizes[max_node];
            for (i = 0; i < f_nodes->size; i++) {
                if (max_prize < prizes[f_nodes->array[i]]) {
                    max_prize = prizes[f_nodes->array[i]];
                    max_node = f_nodes->array[i];
                }
            }
            f_nodes->size = 1;
            f_nodes->array[0] = max_node;
            f_edges->size = 0;
            return true;
        }
    }
    // case 2: there are at least two trees.
    int *sorted_ind = malloc(sizeof(int) * g);
    Tree *trees;
    trees = sort_forest(
            edges, prizes, costs, g, n, f_nodes, f_edges, sorted_ind);
    //clear nodes_f and edges_f, and then update them.
    f_nodes->size = 0, f_edges->size = 0;
    double c_r = C;
    for (i = 0; i < g; i++) {
        int sorted_i = sorted_ind[i];
        double c_tree_i = trees[sorted_i].cost;
        if (c_r >= c_tree_i) {
            c_r -= c_tree_i;
        } else if (c_r > 0.0) {
            // tree_i must have at least two nodes and one edge.
            prune_tree(edges, prizes, costs, n, m, c_r,
                       trees[sorted_i].nodes, trees[sorted_i].edges);
            c_r = 0.0;
        } else {
            // get maximal node
            int max_node = trees[sorted_i].nodes->array[0];
            double max_prize = prizes[max_node];
            for (int ii = 0; ii < trees[sorted_i].nodes->size; ii++) {
                if (max_prize < prizes[trees[sorted_i].nodes->array[ii]]) {
                    max_prize = prizes[trees[sorted_i].nodes->array[ii]];
                    max_node = trees[sorted_i].nodes->array[ii];
                }
            }
            trees[sorted_i].nodes->size = 1;
            trees[sorted_i].nodes->array[0] = max_node;
            trees[sorted_i].edges->size = 0;
        }
        for (j = 0; j < trees[sorted_i].nodes->size; j++) {
            int cur_node = trees[sorted_i].nodes->array[j];
            f_nodes->array[f_nodes->size++] = cur_node;
        }
        for (j = 0; j < trees[sorted_i].edges->size; j++) {
            int cur_edge = trees[sorted_i].edges->array[j];
            f_edges->array[f_edges->size++] = cur_edge;
        }
    }// iterate trees by descending order.
    for (i = 0; i < g; i++) { free(trees[i].nodes), free(trees[i].edges); }
    free(trees), free(sorted_ind);
    printf("pruning forest\n****\n");
    return true;
}


bool head_proj_exact(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double delta, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat) {
    stat->re_nodes->size = 0, stat->re_edges->size = 0,
    stat->run_time = 0.0, stat->num_pcst = 0;
    PCST *pcst;
    clock_t start_time = clock();
    double total_prizes;
    double *tmp_prizes = malloc(sizeof(double) * n);
    // total_prizes will be calculated.
    double pi_min = min_pi(prizes, &total_prizes, n, err_tol, verbose);
    double lambda_r = (2. * C) / (pi_min);
    double lambda_l = 1. / (4. * total_prizes);
    double lambda_m;
    double epsilon_ = (delta * C) / (2. * total_prizes);
    double cost_f;
    int i;
    for (i = 0; i < n; i++) { tmp_prizes[i] = prizes[i] * lambda_r; }
    pcst = make_pcst(edges, tmp_prizes, costs, root, g,
                     epsilon, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
    stat->num_pcst++;
    cost_f = 0.0;
    for (i = 0; i < stat->re_edges->size; i++) {
        cost_f += costs[stat->re_edges->array[i]];
    }
    if (cost_f <= (2. * C)) {
        stat->run_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
        return true;
    }// ensure that we have invariant c(F_r) > 2 C
    while ((lambda_r - lambda_l) > epsilon_) {
        lambda_m = (lambda_l + lambda_r) / 2.;
        for (i = 0; i < n; i++) { tmp_prizes[i] = prizes[i] * lambda_m; }
        pcst = make_pcst(edges, tmp_prizes, costs, root, g,
                         epsilon, pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
        stat->num_pcst++;
        cost_f = 0.0;
        for (i = 0; i < stat->re_edges->size; i++) {
            cost_f += costs[stat->re_edges->array[i]];
        }
        if (cost_f > (2. * C)) {
            lambda_r = lambda_m;
        } else {
            lambda_l = lambda_m;
        }
        if (stat->num_pcst >= max_iter) {
            if (verbose > 0) {
                printf("Warn(head): number iteration is beyond max_iter.\n");
            }
            break;
        }
    } // binary search over the Lagrange parameter lambda
    Array *l_re_nodes = malloc(sizeof(Array));
    Array *l_re_edges = malloc(sizeof(Array));
    l_re_nodes->array = malloc(sizeof(int) * n);
    l_re_edges->array = malloc(sizeof(int) * n);
    for (i = 0; i < n; i++) { tmp_prizes[i] = prizes[i] * lambda_l; }
    pcst = make_pcst(edges, tmp_prizes, costs, root, g,
                     epsilon, pruning, n, m, verbose);
    run_pcst(pcst, l_re_nodes, l_re_edges), free_pcst(pcst);
    stat->num_pcst++;
    Array *r_re_nodes = malloc(sizeof(Array));
    Array *r_re_edges = malloc(sizeof(Array));
    r_re_nodes->array = malloc(sizeof(int) * n);
    r_re_edges->array = malloc(sizeof(int) * n);
    for (i = 0; i < n; i++) { tmp_prizes[i] = prizes[i] * lambda_r; }
    pcst = make_pcst(edges, tmp_prizes, costs, root, g,
                     epsilon, pruning, n, m, verbose);
    run_pcst(pcst, r_re_nodes, r_re_edges), free_pcst(pcst);
    stat->num_pcst++;
    prune_forest(edges, prizes, costs, g, n, m, C, r_re_nodes, r_re_edges);
    double l_prize_f = 0.0, r_prize_f = 0.0;
    for (i = 0; i < l_re_nodes->size; i++) {
        l_prize_f += prizes[l_re_nodes->array[i]];
    }
    for (i = 0; i < r_re_nodes->size; i++) {
        r_prize_f += prizes[r_re_nodes->array[i]];
    }
    if (l_prize_f >= r_prize_f) { //get the left one
        stat->re_nodes->size = l_re_nodes->size;
        for (i = 0; i < stat->re_nodes->size; i++) {
            stat->re_nodes->array[i] = l_re_nodes->array[i];
        }
        stat->re_edges->size = l_re_edges->size;
        for (i = 0; i < stat->re_edges->size; i++) {
            stat->re_edges->array[i] = l_re_edges->array[i];
        }
    } else { // get the right one
        stat->re_nodes->size = r_re_nodes->size;
        for (i = 0; i < stat->re_nodes->size; i++) {
            stat->re_nodes->array[i] = r_re_nodes->array[i];
        }
        stat->re_edges->size = r_re_edges->size;
        for (i = 0; i < stat->re_edges->size; i++) {
            stat->re_edges->array[i] = r_re_edges->array[i];
        }
    }
    stat->run_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(l_re_nodes->array), free(l_re_edges->array);
    free(r_re_nodes->array), free(r_re_edges->array);
    free(l_re_nodes), free(l_re_edges);
    free(r_re_nodes), free(r_re_edges);
    free(tmp_prizes);
    return true;
}

bool head_proj_approx(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double delta, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat) {
    return head_proj_exact(
            edges, costs, prizes, g, C, delta, max_iter, err_tol, root,
            pruning, epsilon, n, m, verbose, stat);
}

bool tail_proj_exact(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double nu, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat) {
    stat->re_nodes->size = 0, stat->re_edges->size = 0,
    stat->run_time = 0.0, stat->num_pcst = 0;
    clock_t start_time = clock();
    int i;
    double total_prizes = 0.0, c_f = 0.0, pi_f_bar = 0.0;
    double pi_min = min_pi(prizes, &total_prizes, n, err_tol, verbose);
    double lambda_0 = pi_min / (2.0 * C);
    double *tmp_costs = malloc(sizeof(double) * m);
    for (i = 0; i < m; i++) { tmp_costs[i] = costs[i] * lambda_0; }
    PCST *pcst;
    pcst = make_pcst(edges, prizes, tmp_costs, root, g,
                     epsilon, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
    stat->num_pcst++;
    for (i = 0; i < stat->re_edges->size; i++) {
        c_f += costs[stat->re_edges->array[i]];
    }
    for (i = 0; i < stat->re_nodes->size; i++) {
        pi_f_bar += prizes[stat->re_nodes->array[i]];
    }
    pi_f_bar = total_prizes - pi_f_bar;
    if ((c_f <= (2.0 * C)) && (pi_f_bar <= 0.0)) {
        free(tmp_costs);
        stat->run_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
        return true;
    }
    double lambda_r = 0., lambda_l = 3. * total_prizes, lambda_m;
    double epsilon_ = (pi_min * fmin(0.5, 1. / nu)) / C;
    while ((lambda_l - lambda_r) > epsilon_) {
        lambda_m = (lambda_l + lambda_r) / 2.;
        for (i = 0; i < m; i++) { tmp_costs[i] = costs[i] * lambda_m; }
        pcst = make_pcst(edges, prizes, tmp_costs, root, g,
                         epsilon, pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
        stat->num_pcst++;
        c_f = 0.0;
        for (i = 0; i < stat->re_edges->size; i++) {
            c_f += costs[stat->re_edges->array[i]];
        }
        if ((c_f >= (2. * C)) && (c_f <= (nu * C))) {
            free(tmp_costs);
            stat->run_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
            return true;
        }
        if (c_f >= (nu * C)) {
            lambda_r = lambda_m;
        } else {
            lambda_l = lambda_m;
        }
        if (stat->num_pcst >= max_iter) {
            if (verbose > 0) {
                printf("Warn(tail): number iteration is beyond max_iter.\n");
            }
            break;
        }
    } // end while
    for (int ii = 0; ii < m; ii++) {
        tmp_costs[ii] = costs[ii] * lambda_l;
    }
    pcst = make_pcst(edges, prizes, tmp_costs, root, g,
                     epsilon, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
    stat->num_pcst++;
    free(tmp_costs);
    stat->run_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    return true;
}


bool tail_proj_approx(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double nu, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat) {
    return tail_proj_exact(
            edges, costs, prizes, g, C, nu, max_iter, err_tol, root, pruning,
            epsilon, n, m, verbose, stat);
}


bool cluster_grid_pcst(
        const EdgePair *edges, const double *costs, const double *prizes,
        int n, int m, int target_num_clusters, double lambda,
        int root, PruningMethod pruning, int verbose,
        GraphStat *stat) {
    double *costs_ = malloc(sizeof(double) * m);
    double *prizes_ = malloc(sizeof(double) * n);
    for (int ii = 0; ii < m; ii++) {
        costs_[ii] = costs[ii] * lambda;
    }
    for (int ii = 0; ii < n; ii++) {
        prizes_[ii] = prizes[ii];
    }
    PCST *pcst;
    pcst = make_pcst(edges, prizes_, costs_, root, target_num_clusters,
                     1e-10, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
    return true;

}

bool cluster_grid_pcst_binsearch(
        const EdgePair *edges, const double *costs, const double *prizes,
        int n, int m, int target_num_clusters, int root, int sparsity_low,
        int sparsity_high, int max_num_iter, PruningMethod pruning,
        int verbose, GraphStat *stat) {
    double *cur_costs = malloc(sizeof(double) * m);
    for (int ii = 0; ii < m; ii++) {
        cur_costs[ii] = costs[ii];
    }


    double *sorted_prizes = malloc(sizeof(double) * n);
    int *sorted_indices = malloc(sizeof(int) * n);
    for (int ii = 0; ii < n; ii++) {
        sorted_prizes[ii] = prizes[ii];
    }
    int guess_pos = n - sparsity_high;
    arg_sort_descend(sorted_prizes, sorted_indices, n);
    double lambda_low = 0.0;
    double lambda_high = 2.0 * sorted_prizes[sorted_indices[guess_pos]];
    bool using_sparsity_low = false;
    bool using_max_value = false;
    if (lambda_high == 0.0) {
        guess_pos = n - sparsity_low;
        lambda_high = 2.0 * sorted_prizes[sorted_indices[guess_pos]];
        if (lambda_high != 0.0) {
            using_sparsity_low = true;
        } else {
            using_max_value = true;
            lambda_high = prizes[0];
            for (int ii = 1; ii < n; ii++) {
                lambda_high = fmax(lambda_high, prizes[ii]);
            }
            lambda_high *= 2.0;
        }
    }
    if (verbose >= 1) {
        const char *sparsity_low_text = "k_low";
        const char *sparsity_high_text = "k_high";
        const char *max_value_text = "max value";
        const char *guess_text = sparsity_high_text;
        if (using_sparsity_low) {
            guess_text = sparsity_low_text;
        } else if (using_max_value) {
            guess_text = max_value_text;
        }
        printf("n = %d  c: %d  k_low: %d  k_high: %d  l_low: %e  l_high: %e  "
               "max_num_iter: %d  (using %s for initial guess).\n",
               n, target_num_clusters, sparsity_low, sparsity_high,
               lambda_low, lambda_high, max_num_iter, guess_text);
    }
    int num_iter = 0;
    lambda_high /= 2.0;
    PCST *pcst;
    int cur_k;
    do {
        num_iter += 1;
        lambda_high *= 2.0;
        for (int ii = 0; ii < m; ++ii) {
            cur_costs[ii] = lambda_high * costs[ii];
        }
        pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters,
                         1e-10, pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
        cur_k = stat->re_nodes->size;
        if (verbose >= 1) {
            printf("increase:   l_high: %e  k: %d\n", lambda_high, cur_k);
        }
    } while (cur_k > sparsity_high && num_iter < max_num_iter);

    if (num_iter < max_num_iter && cur_k >= sparsity_low) {
        if (verbose >= 1) {
            printf("Found good lambda in exponential "
                   "increase phase, returning.\n");
        }
        return true;
    }
    double lambda_mid;
    while (num_iter < max_num_iter) {
        num_iter += 1;
        lambda_mid = (lambda_low + lambda_high) / 2.0;
        for (int ii = 0; ii < m; ++ii) {
            cur_costs[ii] = lambda_mid * costs[ii];
        }
        pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters,
                         1e-10, pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
        cur_k = stat->re_nodes->size;
        if (verbose >= 1) {
            printf("bin_search: l_mid:  %e  k: %d  "
                   "(lambda_low: %e  lambda_high: %e)\n", lambda_mid, cur_k,
                   lambda_low, lambda_high);
        }
        if (cur_k <= sparsity_high && cur_k >= sparsity_low) {
            if (verbose >= 1) {
                printf("Found good lambda in binary "
                       "search phase, returning.\n");
            }
            return true;
        }
        if (cur_k > sparsity_high) {
            lambda_low = lambda_mid;
        } else {
            lambda_high = lambda_mid;
        }
    }
    for (int ii = 0; ii < m; ++ii) {
        cur_costs[ii] = lambda_high * costs[ii];
    }
    pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters,
                     1e-10, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
    if (verbose >= 1) {
        printf("Reached the maximum number of "
               "iterations, using the last l_high: %e  k: %d\n",
               lambda_high, stat->re_nodes->size);
    }
    return true;
}


bool head_tail_binsearch(
        const EdgePair *edges, const double *costs, const double *prizes,
        int n, int m, int target_num_clusters, int root, int sparsity_low,
        int sparsity_high, int max_num_iter, PruningMethod pruning,
        int verbose, GraphStat *stat) {

    // malloc: cur_costs, sorted_prizes, and sorted_indices
    // free: cur_costs, sorted_prizes, and sorted_indices
    double *cur_costs = malloc(sizeof(double) * m);
    double *sorted_prizes = malloc(sizeof(double) * n);
    int *sorted_indices = malloc(sizeof(int) * n);
    for (int ii = 0; ii < m; ii++) {
        cur_costs[ii] = costs[ii];
    }
    for (int ii = 0; ii < n; ii++) {
        sorted_prizes[ii] = prizes[ii];
    }
    int guess_pos = n - sparsity_high;
    arg_sort_descend(sorted_prizes, sorted_indices, n);
    double lambda_low = 0.0;
    double lambda_high = 2.0 * sorted_prizes[sorted_indices[guess_pos]];
    bool using_sparsity_low = false;
    bool using_max_value = false;
    if (lambda_high == 0.0) {
        guess_pos = n - sparsity_low;
        lambda_high = 2.0 * sorted_prizes[sorted_indices[guess_pos]];
        if (lambda_high != 0.0) {
            using_sparsity_low = true;
        } else {
            using_max_value = true;
            lambda_high = prizes[0];
            for (int ii = 1; ii < n; ii++) {
                lambda_high = fmax(lambda_high, prizes[ii]);
            }
            lambda_high *= 2.0;
        }
    }
    if (verbose >= 1) {
        const char *sparsity_low_text = "k_low";
        const char *sparsity_high_text = "k_high";
        const char *max_value_text = "max value";
        const char *guess_text = sparsity_high_text;
        if (using_sparsity_low) {
            guess_text = sparsity_low_text;
        } else if (using_max_value) {
            guess_text = max_value_text;
        }
        printf("n = %d  c: %d  k_low: %d  k_high: %d  l_low: %e  l_high: %e  "
               "max_num_iter: %d  (using %s for initial guess).\n",
               n, target_num_clusters, sparsity_low, sparsity_high,
               lambda_low, lambda_high, max_num_iter, guess_text);
    }
    stat->num_iter = 0;
    lambda_high /= 2.0;
    int cur_k;
    do {
        stat->num_iter += 1;
        lambda_high *= 2.0;
        for (int ii = 0; ii < m; ii++) {
            cur_costs[ii] = lambda_high * costs[ii];
        }
        if (verbose >= 1) {
            for (int ii = 0; ii < m; ii++) {
                printf("E %d %d %.15f\n", edges[ii].first, edges[ii].second,
                       cur_costs[ii]);
            }
            for (int ii = 0; ii < n; ii++) {
                printf("N %d %.15f\n", ii, prizes[ii]);
            }
            printf("\n");
            printf("lambda_high: %f\n", lambda_high);
            printf("target_num_clusters: %d\n", target_num_clusters);
        }
        PCST *pcst = make_pcst(
                edges, prizes, cur_costs, root, target_num_clusters,
                1e-10, pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges);
        free_pcst(pcst);
        cur_k = stat->re_nodes->size;

        if (verbose >= 1) {
            printf("increase:   l_high: %e  k: %d\n", lambda_high, cur_k);
        }
    } while (cur_k > sparsity_high && stat->num_iter < max_num_iter);

    if (stat->num_iter < max_num_iter && cur_k >= sparsity_low) {
        if (verbose >= 1) {
            printf("Found good lambda in exponential "
                   "increase phase, returning.\n");
        }
        free(cur_costs);
        free(sorted_prizes);
        free(sorted_indices);
        return true;
    }
    double lambda_mid;
    while (stat->num_iter < max_num_iter) {
        stat->num_iter += 1;
        lambda_mid = (lambda_low + lambda_high) / 2.0;
        for (int ii = 0; ii < m; ii++) {
            cur_costs[ii] = lambda_mid * costs[ii];
        }
        PCST *pcst = make_pcst(
                edges, prizes, cur_costs, root, target_num_clusters, 1e-10,
                pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges);
        free_pcst(pcst);
        cur_k = stat->re_nodes->size;
        if (verbose >= 1) {
            for (int ii = 0; ii < m; ii++) {
                printf("E %d %d %.15f\n", edges[ii].first, edges[ii].second,
                       cur_costs[ii]);
            }
            for (int ii = 0; ii < n; ii++) {
                printf("N %d %.15f\n", ii, prizes[ii]);
            }
            printf("bin_search: l_mid:  %e  k: %d  "
                   "(lambda_low: %e  lambda_high: %e)\n", lambda_mid, cur_k,
                   lambda_low, lambda_high);
        }
        if (sparsity_low <= cur_k && cur_k <= sparsity_high) {
            if (verbose >= 1) {
                printf("Found good lambda in binary "
                       "search phase, returning.\n");
            }
            free(cur_costs);
            free(sorted_prizes);
            free(sorted_indices);
            return true;
        }
        if (cur_k > sparsity_high) {
            lambda_low = lambda_mid;
        } else {
            lambda_high = lambda_mid;
        }
    }
    for (int ii = 0; ii < m; ++ii) {
        cur_costs[ii] = lambda_high * costs[ii];
    }
    PCST *pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters,
                           1e-10, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges);
    free_pcst(pcst);
    if (verbose >= 1) {
        for (int ii = 0; ii < m; ii++) {
            printf("E %d %d %.15f\n", edges[ii].first, edges[ii].second,
                   cur_costs[ii]);
        }
        printf("\n");
        for (int ii = 0; ii < n; ii++) {
            printf("N %d %.15f\n", ii, prizes[ii]);
        }
        printf("\n");
        printf("Reached the maximum number of "
               "iterations, using the last l_high: %e  k: %d\n",
               lambda_high, stat->re_nodes->size);
    }
    free(cur_costs);
    free(sorted_prizes);
    free(sorted_indices);
    return true;
}