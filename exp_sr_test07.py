# -*- coding: utf-8 -*-

"""
TODO: Please check readme.txt file first!
--
This Python2.7 program is to reproduce Figure-2. In this test, we compare
difference block size b and learning rate eta of our proposed algorithm.
References:
    [1] Nguyen, Nam, Deanna Needell, and Tina Woolf. "Linear convergence of
        stochastic iterative greedy algorithms with sparse constraints."
        IEEE Transactions on Information Theory 63.11 (2017): 6869-6895.
    [2] Hegde, Chinmay, Piotr Indyk, and Ludwig Schmidt. "A nearly-linear time
        framework for graph-structured sparsity." International Conference on
        Machine Learning. 2015.
    [3] Blumensath, Thomas, and Mike E. Davies. "Iterative hard thresholding
        for compressed sensing." Applied and computational harmonic analysis
        27.3 (2009): 265-274.
    [4] Hegde, Chinmay, Piotr Indyk, and Ludwig Schmidt. "Fast recovery from
        a union of subspaces." Advances in Neural Information Processing
        Systems. 2016.
    [5] Lovász, László. "Random walks on graphs: A survey." Combinatorics,
        Paul erdos is eighty 2.1 (1993): 1-46.
    [6] Needell, Deanna, and Joel A. Tropp. "CoSaMP: Iterative signal recovery
        from incomplete and inaccurate samples."
        Applied and computational harmonic analysis 26.3 (2009): 301-321.
"""

import os
import time
import random
import pickle
import multiprocessing
from itertools import product
import numpy as np

try:
    import sparse_module

    try:
        from sparse_module import wrap_head_tail_bisearch
    except ImportError:
        print('cannot find wrap_head_tail_bisearch method in sparse_module')
        sparse_module = None
        exit(0)
except ImportError:
    print('\n'.join([
        'cannot find the module: sparse_module',
        'try run: \'python setup.py build_ext --inplace\' first! ']))


def algo_head_tail_bisearch(
        edges, x, costs, g, root, s_low, s_high, max_num_iter, verbose):
    """ This is the wrapper of head/tail-projection proposed in [2].
    :param edges:           edges in the graph.
    :param x:               projection vector x.
    :param costs:           edge costs in the graph.
    :param g:               the number of connected components.
    :param root:            root of subgraph. Usually, set to -1: no root.
    :param s_low:           the lower bound of the sparsity.
    :param s_high:          the upper bound of the sparsity.
    :param max_num_iter:    the maximum number of iterations used in
                            binary search procedure.
    :param verbose: print out some information.
    :return:            1.  the support of the projected vector
                        2.  the projected vector
    """
    prizes = x * x
    # to avoid too large upper bound problem.
    if s_high >= len(prizes) - 1:
        s_high = len(prizes) - 1
    re_nodes = wrap_head_tail_bisearch(
        edges, prizes, costs, g, root, s_low, s_high, max_num_iter, verbose)
    proj_w = np.zeros_like(x)
    proj_w[re_nodes[0]] = x[re_nodes[0]]
    return re_nodes[0], proj_w


def simu_grid_graph(width, height, rand_weight=False):
    """ Generate a grid graph with size, width x height. Totally there will be
        width x height number of nodes in this generated graph.
    :param width:       the width of the grid graph.
    :param height:      the height of the grid graph.
    :param rand_weight: the edge costs in this generated grid graph.
    :return:            1.  list of edges
                        2.  list of edge costs
    """
    np.random.seed()
    if width < 0 and height < 0:
        print('Error: width and height should be positive.')
        return [], []
    width, height = int(width), int(height)
    edges, weights = [], []
    index = 0
    for i in range(height):
        for j in range(width):
            if (index % width) != (width - 1):
                edges.append((index, index + 1))
                if index + width < int(width * height):
                    edges.append((index, index + width))
            else:
                if index + width < int(width * height):
                    edges.append((index, index + width))
            index += 1
    edges = np.asarray(edges, dtype=int)
    # random generate costs of the graph
    if rand_weight:
        weights = []
        while len(weights) < len(edges):
            weights.append(random.uniform(1., 2.0))
        weights = np.asarray(weights, dtype=np.float64)
    else:  # set unit weights for edge costs.
        weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights


def sensing_matrix(n, x, norm_noise=0.0):
    """ Generate sensing matrix (design matrix). This generated sensing
        matrix is a Gaussian matrix, i.e., each entry ~ N(0,\sigma/\sqrt(n)).
        Please see more details in equation (1.2) shown in reference [6].
    :param n:           the number of measurements required.
    :param x:           the input signal.
    :param norm_noise:  plus ||norm_noise|| noise on the measurements.
    :return:            1.  the design matrix
                        2.  the vector of measurements
                        3.  the noised vector.
    """
    p = len(x)
    x_mat = np.random.normal(loc=0.0, scale=1.0, size=(n * p)) / np.sqrt(n)
    x_mat = x_mat.reshape((n, p))
    y_tr = np.dot(x_mat, x)
    noise_e = np.random.normal(loc=0.0, scale=1.0, size=len(y_tr))
    y_e = y_tr + (norm_noise / np.linalg.norm(noise_e)) * noise_e
    return x_mat, y_tr, y_e


def random_walk(edges, s, init_node=None, restart=0.0):
    """ The random walk on graphs. Please see details in reference [5].
    :param edges:       the edge list of the graph.
    :param s:           the sparsity ( number of nodes) in the true subgraph.
    :param init_node:   initial point of the random walk.
    :param restart:     with restart.
    :return:            1. list of nodes walked.
                        2. list of edges walked.
    """
    np.random.seed()
    adj, nodes = dict(), set()
    for edge in edges:  # construct the adjacency matrix.
        uu, vv = int(edge[0]), int(edge[1])
        nodes.add(uu)
        nodes.add(vv)
        if uu not in adj:
            adj[uu] = set()
        adj[uu].add(vv)
        if vv not in adj:
            adj[vv] = set()
        adj[vv].add(uu)
    if init_node is None:
        # random select an initial node.
        rand_start_point = random.choice(list(nodes))
        init_node = list(adj.keys())[rand_start_point]
    if init_node not in nodes:
        print('Error: the initial_node is not in the graph!')
        return [], []
    if not (0.0 <= restart < 1.0):
        print('Error: the restart probability not in (0.0,1.0)')
        return [], []
    if not (0 <= s <= len(nodes)):
        print('Error: the number of nodes not in [0,%d]' % len(nodes))
        return [], []
    subgraph_nodes, subgraph_edges = set(), set()
    next_node = init_node
    subgraph_nodes.add(init_node)
    if s <= 1:
        return subgraph_nodes, subgraph_edges
    # get a connected subgraph with s nodes.
    while len(subgraph_nodes) < s:
        next_neighbors = list(adj[next_node])
        rand_nei = random.choice(next_neighbors)
        subgraph_nodes.add(rand_nei)
        subgraph_edges.add((next_node, rand_nei))
        subgraph_edges.add((rand_nei, next_node))
        next_node = rand_nei  # go to next node.
        if random.random() < restart:
            next_node = init_node
    return list(subgraph_nodes), list(subgraph_edges)


def algo_sto_iht(x_mat, y_tr, max_epochs, lr, s, x_star, x0, tol_algo, b):
    """ Stochastic Iterative Hard Thresholding Method proposed in [1].
    :param x_mat:       the design matrix.
    :param y_tr:        the array of measurements.
    :param max_epochs:  the maximum epochs (iterations) allowed.
    :param lr:          the learning rate (should be 1.0).
    :param s:           the sparsity parameter.
    :param x_star:      the true signal.
    :param x0:          x0 is the initial point.
    :param tol_algo:    tolerance parameter for early stopping.
    :param b:           block size
    :return:            1.  the final estimation error,
                        2.  number of epochs(iterations) used,
                        3.  and the run time.
    """
    np.random.seed()
    start_time = time.time()
    x_hat = x0
    (n, p) = x_mat.shape
    x_tr_t = np.transpose(x_mat)
    b = n if n < b else b
    num_blocks = int(n) / int(b)
    prob = [1. / num_blocks] * num_blocks
    num_epochs = 0
    num_iterations = 0
    run_time_proj = 0
    for epoch_i in range(max_epochs):
        num_epochs += 1
        for _ in range(num_blocks):
            ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            xtx = np.dot(x_tr_t[:, block], x_mat[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            gradient = - 2. * (xty - np.dot(xtx, x_hat))
            bt = x_hat - (lr / (prob[ii] * num_blocks)) * gradient
            start_time_proj = time.time()
            bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
            run_time_proj += time.time() - start_time_proj
            x_hat = bt
            num_iterations += 1
            # early stopping for diverge cases due to the large learning rate
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    err = np.linalg.norm(x_star - x_hat)
    return num_epochs, num_iterations, run_time, run_time_proj, err


def algo_graph_sto_iht(
        x_mat, y_tr, max_epochs, lr, x_star, x0, tol_algo, edges, costs, s, b,
        g=1, root=-1, gamma=0.1, proj_max_num_iter=50, verbose=0):
    """ Graph Stochastic Iterative Hard Thresholding.
    :param x_mat:       the design matrix.
    :param y_tr:        the array of measurements.
    :param max_epochs:  the maximum epochs (iterations) allowed.
    :param lr:          the learning rate (should be 1.0).
    :param x_star:      the true signal.
    :param x0:          x0 is the initial point.
    :param tol_algo:    tolerance parameter for early stopping.
    :param edges:       edges in the graph.
    :param costs:       edge costs
    :param s:           sparsity
    :param b: the block size
    :param g:           number of connected component in the true signal.
    :param root:        the root included in the result (default -1: no root).
    :param gamma:       to control the upper bound of sparsity.
    :param proj_max_num_iter: maximum number of iterations of projection.
    :param verbose: print out some information.
    :return:            1.  the final estimation error,
                        2.  number of epochs(iterations) used,
                        3.  and the run time.
    """
    np.random.seed()
    start_time = time.time()
    x_hat = np.copy(x0)
    x_tr_t = np.transpose(x_mat)

    # graph projection para
    h_low = int(len(x0) / 2)
    h_high = int(h_low * (1. + gamma))
    t_low = int(s)
    t_high = int(s * (1. + gamma))

    (n, p) = x_mat.shape
    # if block size is larger than n,
    # just treat it as a single block (batch)
    b = n if n < b else b
    num_blocks = int(n) / int(b)
    prob = [1. / num_blocks] * num_blocks

    x_err_list = []
    x_iter_err_list = []
    run_time_head = 0.0
    run_time_tail = 0.0
    num_iterations = 0
    num_epochs = 0
    for epoch_i in range(max_epochs):
        num_epochs += 1
        for _ in range(num_blocks):
            ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            xtx = np.dot(x_tr_t[:, block], x_mat[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            gradient = -2. * (xty - np.dot(xtx, x_hat))
            start_time_head = time.time()
            head_nodes, proj_grad = algo_head_tail_bisearch(
                edges, gradient, costs, g, root, h_low, h_high,
                proj_max_num_iter, verbose)
            run_time_head += time.time() - start_time_head

            bt = x_hat - (lr / (prob[ii] * num_blocks)) * proj_grad
            start_time_tail = time.time()
            tail_nodes, proj_bt = algo_head_tail_bisearch(
                edges, bt, costs, g, root,
                t_low, t_high, proj_max_num_iter, verbose)
            run_time_tail += time.time() - start_time_tail
            x_hat = proj_bt
            x_iter_err_list.append(np.linalg.norm(x_hat - x_star))
            num_iterations += 1

        x_err_list.append(np.linalg.norm(x_hat - x_star))
        # early stopping for diverge cases due to the large learning rate
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    err = np.linalg.norm(x_star - x_hat)
    run_time_proj = run_time_head + run_time_tail
    return num_epochs, num_iterations, run_time, run_time_proj, err


def run_single_test_diff_b(data):
    np.random.seed()
    # generate Gaussian measurement matrix.
    # each entry is generated from N(0,1)/sqrt(n) Gaussian.
    s, n, p, b = data['s'], data['n'], data['p'], data['b']
    lr = data['lr'][b]
    x0 = data['x0']
    x_star = data['x_star']
    trial_i = data['trial_i']
    tol_algo = data['tol_algo']
    max_epochs = data['max_epochs']
    x_mat, y_tr, _ = sensing_matrix(n=n, x=data['x_star'])

    edges = data['proj_para']['edges']
    costs = data['proj_para']['costs']

    metric_list = []
    # ------------- StoIHT --------
    num_epochs, num_iterations, run_time, run_time_proj, err = algo_sto_iht(
        x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs, lr=lr, s=s,
        x_star=x_star, x0=x0, tol_algo=tol_algo, b=b)
    metric_list.append(('sto-iht', num_epochs,
                        run_time, num_iterations, run_time_proj, err))
    # ------------- GraphStoIHT --------
    num_epochs, num_iterations, run_time, run_time_proj, err = \
        algo_graph_sto_iht(x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs,
                           lr=lr, x_star=x_star, x0=x0, tol_algo=tol_algo,
                           edges=edges, costs=costs, s=s, b=b)
    metric_list.append(('graph-sto-iht', num_epochs,
                        run_time, num_iterations, run_time_proj, err))
    return trial_i, b, metric_list


def run_test_diff_b(
        s, p, height, width, max_epochs, total_samples, tol_algo, tol_rec,
        b_list, trim_ratio, num_cpus, num_trials):
    np.random.seed()
    start_time = time.time()
    input_data_list = []
    saved_data = dict()
    for (b, trial_i) in product(b_list, range(num_trials)):
        edges, costs = simu_grid_graph(height=height, width=width)
        init_node = (height / 2) * width + height / 2
        sub_graphs = {_: random_walk(edges=edges, s=_, init_node=init_node)
                      for _ in [s]}
        x_star = np.zeros(p)  # using standard Gaussian signal.
        x_star[sub_graphs[s][0]] = np.random.normal(loc=0.0, scale=1.0, size=s)
        data = {
            # we need to keep the consistency with Needell's code
            # when b==180 corresponding to batched-versions.
            'lr': {b: 1. if b != total_samples else 1. / 2. for b in b_list},
            'max_epochs': max_epochs,
            'trial_i': trial_i,
            's': s,
            'n': total_samples,
            's_list': [s],
            'p': p,
            'b': b,
            'x_star': x_star,
            'x0': np.zeros(p),
            'subgraph': sub_graphs[s][0],
            'tol_algo': tol_algo,
            'height': height,
            'width': width,
            'tol_rec': tol_rec,
            'subgraph_edges': sub_graphs[s][1],
            'verbose': 0,
            'proj_para': {'edges': edges, 'costs': costs}}
        if s not in saved_data:
            saved_data[s] = data
        input_data_list.append(data)
    pool = multiprocessing.Pool(processes=num_cpus)
    results_pool = pool.map(run_single_test_diff_b, input_data_list)
    pool.close()
    pool.join()
    for i, metric in zip(range(5), ['num_epochs', 'run_time', 'num_iterations',
                                    'run_time_proj', 'error']):
        aver_results = {'sto-iht': {b: [] for b in b_list},
                        'graph-sto-iht': {b: [] for b in b_list}}
        for trial_i, b, re in results_pool:
            for _ in re:
                method, num_epochs, num_iterations, run_time, run_time_proj, err = _
                xx = num_epochs, num_iterations, run_time, run_time_proj, err
                aver_results[method][b].append(xx[i])
        print(metric)
        for b in b_list:
            print(b, np.mean(sorted(aver_results['sto-iht'][b])),
                  np.mean(sorted(aver_results['graph-sto-iht'][b])))


def main():
    # try 50 different trials and take average on 44 trials.
    num_trials = 50
    # tolerance of the algorithm
    tol_algo = 1e-7
    # tolerance of the recovery.
    tol_rec = 1e-6
    # the dimension of the grid graph.
    p = 400
    # the trimmed ratio ( about 5% of the best and worst have been removed).
    trim_ratio = 0.05
    # height and width of the grid graph.
    height, width = 20, 20
    s = 20
    total_samples = 400
    b_list = []
    for i in range(1, 15):
        b_list.append(int((1. * p) / (1. * i)))
    root_p = 'results/'
    if not os.path.exists(root_p):
        os.mkdir(root_p)
    num_cpus = int(os.sys.argv[1])
    run_test_diff_b(s=s,
                    p=p,
                    height=height,
                    width=width,
                    max_epochs=100,
                    total_samples=total_samples,
                    tol_algo=tol_algo,
                    tol_rec=tol_rec,
                    b_list=b_list,
                    trim_ratio=trim_ratio,
                    num_cpus=num_cpus,
                    num_trials=num_trials)


def single_test_1():
    height, width = 80, 80
    s = 100
    lr = .5
    n = 1000
    start_time = time.time()
    edges, costs = simu_grid_graph(height=height, width=width)
    init_node = (height / 2) * width + height / 2
    sub_graphs = {_: random_walk(edges=edges, s=_, init_node=init_node)
                  for _ in [s]}
    x_star = np.zeros(height * width)  # using standard Gaussian signal.
    x_star[sub_graphs[s][0]] = np.random.normal(loc=0.0, scale=1.0, size=s)

    x_mat, y_tr, _ = sensing_matrix(n=n, x=x_star)
    x0 = np.zeros(height * width)
    x_star = x_star
    tol_algo = 1e-7
    max_epochs = 500
    for b in [1000, 500, 250, 200, 100]:
        num_epochs, num_iterations, run_time, run_time_proj, err = algo_graph_sto_iht(
            x_mat, y_tr, max_epochs, lr, x_star, x0, tol_algo, edges, costs, s,
            b, g=1, root=-1, gamma=0.1, proj_max_num_iter=50, verbose=0)
        print('num_epochs, num_iterations, run_time, run_time_proj')
        print(num_epochs, num_iterations, run_time, run_time_proj)


def single_test_2():
    height, width = 80, 80
    s = 100
    lr = 0.5
    n = 2000
    start_time = time.time()
    edges, costs = simu_grid_graph(height=height, width=width)
    init_node = (height / 2) * width + height / 2
    sub_graphs = {_: random_walk(edges=edges, s=_, init_node=init_node)
                  for _ in [s]}
    x_star = np.zeros(height * width)  # using standard Gaussian signal.
    x_star[sub_graphs[s][0]] = np.random.normal(loc=0.0, scale=1.0, size=s)

    x_mat, y_tr, _ = sensing_matrix(n=n, x=x_star)
    x0 = np.zeros(height * width)
    x_star = x_star
    tol_algo = 1e-7
    max_epochs = 500
    for b in [2000, 1000, 500, 250, 200, 100]:
        num_epochs, num_iterations, run_time, run_time_proj, err = algo_sto_iht(
            x_mat, y_tr, max_epochs, lr, s, x_star, x0, tol_algo, b)
        print('num_epochs, num_iterations, run_time, run_time_proj')
        print(num_epochs, num_iterations, run_time, run_time_proj)
        num_epochs, num_iterations, run_time, run_time_proj, err = algo_graph_sto_iht(
            x_mat, y_tr, max_epochs, lr, x_star, x0, tol_algo, edges, costs, s,
            b, g=1, root=-1, gamma=0.1, proj_max_num_iter=50, verbose=0)
        print('num_epochs, num_iterations, run_time, run_time_proj')
        print(num_epochs, num_iterations, run_time, run_time_proj)


def single_test_3():
    height, width = 80, 80
    s = 100
    lr = 0.5
    n = 2000
    start_time = time.time()
    edges, costs = simu_grid_graph(height=height, width=width)
    init_node = (height / 2) * width + height / 2
    sub_graphs = {_: random_walk(edges=edges, s=_, init_node=init_node)
                  for _ in [s]}
    x_star = np.zeros(height * width)  # using standard Gaussian signal.
    x_star[sub_graphs[s][0]] = np.random.normal(loc=0.0, scale=1.0, size=s)
    x_mat, y_tr, _ = sensing_matrix(n=n, x=x_star)
    x0 = np.zeros(height * width)
    x_star = x_star
    tol_algo = 1e-7
    max_epochs = 500
    for b in [2000]:
        num_epochs, num_iterations, run_time, run_time_proj, err = algo_sto_iht(
            x_mat, y_tr, max_epochs, lr, s, x_star, x0, tol_algo, b)
        print('num_epochs, num_iterations, run_time, run_time_proj')
        print(num_epochs, num_iterations, run_time, run_time_proj)


if __name__ == '__main__':
    main()
