# -*- coding: utf-8 -*-

"""
In this test, we compare GraphStoIHT with three baseline methods including
IHT, StoIHT, and GraphIHT. IHT is proposed in [3]. StoIHT is proposed in [1].
GraphIHT is proposed [4] with head/tail projections in [2].

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

# TODO You need to:
    1.  install numpy, matplotlib (optional), and networkx (optional).
    2.  build our sparse_module by executing ./build.sh please check our
        readme.md file. If you do not know how to compile this library.
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


def algo_iht(x_mat, y_tr, max_epochs, lr, s, x_star, x0, tol_algo):
    """ Iterative Hard Thresholding Method proposed in reference [3]. The
        standard iterative hard thresholding method for compressive sensing.
    :param x_mat:       the design matrix.
    :param y_tr:        the array of measurements.
    :param max_epochs:  the maximum epochs (iterations) allowed.
    :param lr:          the learning rate (should be 1.0).
    :param s:           the sparsity parameter.
    :param x_star:      the true signal.
    :param x0:          x0 is the initial point.
    :param tol_algo:    tolerance parameter for early stopping.
    :return:            1.  the final estimation error,
                        2.  number of epochs(iterations) used,
                        3.  and the run time.
    """
    start_time = time.time()
    x_hat = x0
    (n, p) = x_mat.shape
    x_tr_t = np.transpose(x_mat)
    xtx = np.dot(x_tr_t, x_mat)
    xty = np.dot(x_tr_t, y_tr)

    num_epochs = 0
    for epoch_i in range(max_epochs):
        num_epochs += 1
        bt = x_hat - lr * (np.dot(xtx, x_hat) - xty)
        bt[np.argsort(np.abs(bt))[0:p - s]] = 0.  # thresholding step
        x_hat = bt

        # early stopping for diverge cases due to the large learning rate
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
    x_err = np.linalg.norm(x_hat - x_star)
    run_time = time.time() - start_time
    return x_err, num_epochs, run_time


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
    for epoch_i in range(max_epochs):
        num_epochs += 1
        for _ in range(num_blocks):
            ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            xtx = np.dot(x_tr_t[:, block], x_mat[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            gradient = - 2. * (xty - np.dot(xtx, x_hat))
            bt = x_hat - (lr / (prob[ii] * num_blocks)) * gradient
            bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
            x_hat = bt

        # early stopping for diverge cases due to the large learning rate
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break

    x_err = np.linalg.norm(x_hat - x_star)
    run_time = time.time() - start_time
    return x_err, num_epochs, run_time


def algo_graph_iht(
        x_mat, y_tr, max_epochs, lr, x_star, x0, tol_algo, edges, costs, s,
        g=1, root=-1, gamma=0.1, proj_max_num_iter=50, verbose=0):
    """ Graph Iterative Hard Thresholding proposed in [4] and projection
        operator is proposed in [2].
    :param x_mat:       the design matrix.
    :param y_tr:        the array of measurements.
    :param max_epochs:  the maximum epochs (iterations) allowed.
    :param lr:          the learning rate (should be 1.0).
    :param x_star:      x_star is the true signal.
    :param x0:          x0 is the initial point.
    :param tol_algo:    tolerance parameter for early stopping.
    :param edges:       edges in the graph.
    :param costs:       edge costs
    :param s:           sparsity
    :param g:           number of connected component in the true signal.
    :param root:        the root included in the result (default -1: no root).
    :param gamma:       to control the upper bound of sparsity.
    :param proj_max_num_iter: maximum number of iterations of projection.
    :param verbose:     print out some information.
    :return:            1.  the final estimation error,
                        2.  number of epochs(iterations) used,
                        3.  and the run time.
    """
    start_time = time.time()
    x_hat = np.copy(x0)
    xtx = np.dot(np.transpose(x_mat), x_mat)
    xty = np.dot(np.transpose(x_mat), y_tr)

    # graph projection para
    h_low = int(len(x0) / 2)
    h_high = int(h_low * (1. + gamma))
    t_low = int(s)
    t_high = int(s * (1. + gamma))

    num_epochs = 0
    for epoch_i in range(max_epochs):
        num_epochs += 1
        grad = -1. * (xty - np.dot(xtx, x_hat))
        head_nodes, proj_gradient = algo_head_tail_bisearch(
            edges, grad, costs, g, root, h_low, h_high,
            proj_max_num_iter, verbose)
        bt = x_hat - lr * proj_gradient
        tail_nodes, proj_bt = algo_head_tail_bisearch(
            edges, bt, costs, g, root, t_low, t_high,
            proj_max_num_iter, verbose)
        x_hat = proj_bt

        # early stopping for diverge cases due to the large learning rate
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
    x_err = np.linalg.norm(x_hat - x_star)
    run_time = time.time() - start_time
    return x_err, num_epochs, run_time


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

    num_epochs = 0
    for epoch_i in range(max_epochs):
        num_epochs += 1
        for _ in range(num_blocks):
            ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            xtx = np.dot(x_tr_t[:, block], x_mat[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            gradient = -2. * (xty - np.dot(xtx, x_hat))
            head_nodes, proj_grad = algo_head_tail_bisearch(
                edges, gradient, costs, g, root, h_low, h_high,
                proj_max_num_iter, verbose)
            bt = x_hat - (lr / (prob[ii] * num_blocks)) * proj_grad
            tail_nodes, proj_bt = algo_head_tail_bisearch(
                edges, bt, costs, g, root,
                t_low, t_high, proj_max_num_iter, verbose)
            x_hat = proj_bt

        # early stopping for diverge cases due to the large learning rate
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
    x_err = np.linalg.norm(x_hat - x_star)
    run_time = time.time() - start_time
    return x_err, num_epochs, run_time


def print_helper(method, trial_i, s, n, num_epochs, err, run_time):
    print('%13s trial_%03d s: %02d n: %03d epochs: %03d '
          'rec_error: %.4e run_time: %.4e' %
          (method, trial_i, s, n, num_epochs, err, run_time))


def run_single_test(data):
    np.random.seed()
    s, n, p, b = data['s'], data['n'], data['p'], data['b']
    lr = data['lr']
    x0 = data['x0']
    x_star = data['x_star']
    trial_i = data['trial_i']
    tol_algo = data['tol_algo']
    max_epochs = data['max_epochs']
    x_mat, y_tr, _ = sensing_matrix(n=n, x=data['x_star'])

    edges = data['proj_para']['edges']
    costs = data['proj_para']['costs']

    rec_error = []
    # ------------- IHT ----------------
    err, num_epochs, run_time = algo_iht(
        x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs, lr=lr, s=s,
        x_star=x_star, x0=x0, tol_algo=tol_algo)
    rec_error.append(('iht', err))
    print_helper('iht', trial_i, s, n, num_epochs, err, run_time)

    # ------------- StoIHT -------------
    err, num_epochs, run_time = algo_sto_iht(
        x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs, lr=lr, s=s,
        x_star=x_star, x0=x0, tol_algo=tol_algo, b=b)
    rec_error.append(('sto-iht', err))
    print_helper('sto-iht', trial_i, s, n, num_epochs, err, run_time)

    # ------------- GraphIHT -----------
    err, num_epochs, run_time = algo_graph_iht(
        x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs, lr=lr, x_star=x_star,
        x0=x0, tol_algo=tol_algo, edges=edges, costs=costs, s=s)
    rec_error.append(('graph-iht', err))
    print_helper('graph-iht', trial_i, s, n, num_epochs, err, run_time)

    # ------------- GraphStoIHT --------
    err, num_epochs, run_time = algo_graph_sto_iht(
        x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs, lr=lr, x_star=x_star,
        x0=x0, tol_algo=tol_algo, edges=edges, costs=costs, s=s, b=b)
    rec_error.append(('graph-sto-iht', err))
    print_helper('graph-sto-iht', trial_i, s, n, num_epochs, err, run_time)
    return trial_i, n, s, rec_error


def run_test(p, lr, height, max_epochs, width, tol_algo, tol_rec, s_list,
             n_list, trim_ratio, num_cpus, num_trials, method_list,
             save_data_path):
    np.random.seed()
    start_time = time.time()
    input_data_list = []
    saved_data = dict()

    for (s, n, trial_i) in product(s_list, n_list, range(num_trials)):
        print('data pair: (trial_%03d, s: %02d, n: %03d)' % (trial_i, s, n))
        b = int(np.fmin(s, n))
        edges, costs = simu_grid_graph(height=height, width=width)
        # initial node is located in the center of the grid graph.
        init_node = (height / 2) * width + height / 2
        sub_graphs = {s: random_walk(edges, s, init_node, 0.) for s in s_list}
        x_star = np.zeros(p)  # using standard Gaussian signal.
        x_star[sub_graphs[s][0]] = np.random.normal(loc=0.0, scale=1.0, size=s)
        data = {'lr': lr,
                'max_epochs': max_epochs,
                'trial_i': trial_i,
                's': s,
                'n': n,
                'n_list': n_list,
                's_list': s_list,
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
                # parameters used in head and tail projection.
                'proj_para': {'edges': edges, 'costs': costs}}
        if s not in saved_data:
            saved_data[s] = data
        input_data_list.append(data)
    pool = multiprocessing.Pool(processes=num_cpus)
    results_pool = pool.map(run_single_test, input_data_list)
    pool.close()
    pool.join()
    sum_results = {
        method: {s: np.zeros((num_trials, len(n_list))) for s in s_list}
        for method in method_list}
    for trial_i, n, s, re in results_pool:
        n_ind = list(n_list).index(n)
        for method, val in re:
            sum_results[method][s][trial_i][n_ind] = val

    # try to trim 5% of the results (rounding when necessary).
    num_trim = int(round(trim_ratio * num_trials))
    trim_results = {
        method: {s: np.zeros(shape=(num_trials - 2 * num_trim, len(n_list)))
                 for s in s_list} for method in method_list}
    for method, s in product(method_list, s_list):
        re = sum_results[method][s]
        # remove 5% best and 5% worst.
        trimmed_re = np.sort(re, axis=0)[num_trim:num_trials - num_trim, :]
        trim_results[method][s] = trimmed_re
    for method in method_list:
        for s in s_list:
            re = trim_results[method][s]
            re[re > tol_rec] = 0.
            # cases that successfully recovered.
            re[re != 0.0] = 1.0
            trim_results[method][s] = re

    print('save results to file: %s' % save_data_path)
    pickle.dump({'trim_results': trim_results,
                 'sum_results': sum_results,
                 'saved_data': saved_data}, open(save_data_path, 'wb'))
    print('total run time of %02d trials: %.2f seconds.' %
          (num_trials, time.time() - start_time))


def show_test(s_list, n_list, method_list, label_list, save_data_path):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    rc('text', usetex=True)

    rcParams['figure.figsize'] = 8, 6
    color_list = ['b', 'g', 'm', 'r']
    marker_list = ['X', 'o', 'P', 's']
    results = pickle.load(open(save_data_path))['trim_results']
    fig, ax = plt.subplots(2, 2, sharex='all', sharey='all')
    for ii, jj in product(range(2), range(2)):
        ax[ii, jj].grid(b=True, which='both', color='gray',
                        linestyle='dotted', axis='both')
        ax[ii, jj].spines['right'].set_visible(False)
        ax[ii, jj].spines['top'].set_visible(False)

    ax[1, 0].set_xticks(np.arange(0, max(n_list) + 1, 50))
    ax[1, 1].set_xticks(np.arange(0, max(n_list) + 1, 50))

    ax[0, 0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1, 0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for s in s_list:
        print(' '.join(method_list))
        re_mat = np.zeros(shape=(len(n_list), 4))
        for method_ind, method in enumerate(method_list):
            for ind, _ in enumerate(
                    np.mean(results[method_list[method_ind]][s], axis=0)):
                re_mat[ind][method_ind] = _
        for ind, _ in enumerate(n_list):
            row = [str(_)]
            row.extend([str('%.3f' % _) for _ in re_mat[ind]])
            print(', '.join(row))
    for m_ind, s in enumerate(s_list):
        ii, jj = m_ind / 2, m_ind % 2
        for method_ind, method in enumerate(method_list):
            re = np.mean(results[method_list[method_ind]][s], axis=0)
            ax[ii, jj].plot(n_list, re, c=color_list[method_ind],
                            markerfacecolor='none',
                            linestyle='-', marker=marker_list[method_ind],
                            markersize=6., markeredgewidth=1.,
                            linewidth=1.5, label=label_list[method_ind])
        ax[ii, jj].set_title(r"$\displaystyle s=%d$" % s)
        ttl = ax[ii, jj].title
        ttl.set_position([.5, 0.97])

    for i in range(2):
        ax[1, i].set_xlabel(r"$\displaystyle n$", labelpad=-0.5)
        ax[i, 0].set_ylabel(r"Probability of Recovery")
    ax[1, 1].legend(loc='center right', framealpha=1.,
                    bbox_to_anchor=(0.55, 0.5),
                    fontsize=14., frameon=True, borderpad=0.1,
                    labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    plt.subplots_adjust(wspace=0.0, hspace=0.2)
    save_data_path = save_data_path.replace('pkl', 'pdf')
    print('save fig to: %s' % save_data_path)
    plt.savefig(save_data_path, dpi=600, bbox_inches='tight', pad_inches=0,
                format='pdf')
    plt.close()


def generate_figures(root_p, save_data_path):
    import networkx as nx
    import matplotlib.pyplot as plt

    data = pickle.load(open(save_data_path))['saved_data']
    edges = data[20]['proj_para']['edges']
    height, width = data[20]['height'], data[20]['width']

    p = data[20]['p']
    plt.figure(figsize=(1.5, 1.5))
    for s in data:
        pos, graph = dict(), nx.Graph()
        black_edges = []
        red_edges = []
        red_edge_list = []
        for edge in edges:
            graph.add_edge(edge[0], edge[1])
            if (edge[0], edge[1]) in data[s]['subgraph_edges']:
                red_edges.append('r')
                red_edge_list.append((edge[0], edge[1]))
            else:
                black_edges.append('k')
        for index, (i, j) in enumerate(product(range(height), range(width))):
            graph.add_node(index)
            pos[index] = (j, height - i)

        print('generate subgraph, which has %02d nodes.' % s)
        nx.draw_networkx_nodes(
            graph, pos, node_size=15, nodelist=range(p), linewidths=.5,
            node_color='w', edgecolors='k', font_size=6)
        x_values = np.random.normal(loc=0.0, scale=1.0,
                                    size=len(data[s]['subgraph']))
        nx.draw_networkx_nodes(
            graph, pos, node_size=15, nodelist=data[s]['subgraph'],
            linewidths=.5, node_color=x_values, cmap='jet',
            edgecolors='k', font_size=6)
        nx.draw_networkx_edges(
            graph, pos, alpha=0.8, width=0.8, edge_color='k', font_size=6)
        nx.draw_networkx_edges(
            graph, pos, alpha=0.7, width=2.0, edgelist=red_edge_list,
            edge_color='r', font_size=6)
        plt.axis('off')
        fig = plt.gcf()
        fig.set_figheight(1.5)
        fig.set_figwidth(1.5)
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        for ax in fig.axes:
            ax.axis('off')
            ax.margins(0.02, 0.02)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
        f_name = root_p + 'results_exp_sr_test01_s_%02d.pdf' % s
        fig.savefig(f_name, dpi=600, pad_inches=0.0, format='pdf')
        plt.close()


def main():
    # list of methods considered
    method_list = ['iht', 'sto-iht', 'graph-iht', 'graph-sto-iht']
    label_list = ['IHT', 'StoIHT', 'GraphIHT', 'GraphStoIHT']
    # 4 different sparsity parameters considered.
    s_list = np.asarray([8, 20, 28, 36])
    # number of measurements list
    n_list = np.arange(5, 251, 5)
    # try 50 different trials
    num_trials = 50
    # tolerance of the algorithm
    tol_algo = 1e-7
    # tolerance of the recovery.
    tol_rec = 1e-6
    # the dimension of the grid graph.
    p = 256
    # the trimmed ratio (5% of the best and worst have been removed).
    trim_ratio = 0.05
    # height and width of the grid graph.
    height, width = 16, 16
    # maximum number of epochs allowed for all methods.
    max_epochs = 500
    # learning rate ( consistent with Needell's paper)
    lr = 1.0

    # TODO config the path by yourself.
    root_p = 'results/'
    if not os.path.exists(root_p):
        os.mkdir(root_p)
    save_data_path = root_p + 'results_exp_sr_test01.pkl'

    if len(os.sys.argv) <= 1:
        print('\n'.join(['please use one of the following commands: ',
                         '1. python exp_sr_test01.py run_test 50',
                         '2. python exp_sr_test01.py show_test',
                         '3. python exp_sr_test01.py gen_figures']))
        exit(0)
    command = os.sys.argv[1]
    if command == 'run_test':
        num_cpus = int(os.sys.argv[2])
        run_test(p=p,
                 lr=lr,
                 height=height,
                 width=width,
                 max_epochs=max_epochs,
                 tol_algo=tol_algo,
                 tol_rec=tol_rec,
                 s_list=s_list,
                 n_list=n_list,
                 trim_ratio=trim_ratio,
                 num_cpus=num_cpus,
                 num_trials=num_trials,
                 method_list=method_list,
                 save_data_path=save_data_path)
    elif command == 'show_test':
        show_test(s_list=s_list,
                  n_list=n_list,
                  method_list=method_list,
                  label_list=label_list,
                  save_data_path=save_data_path)
    elif command == 'gen_figures':
        generate_figures(root_p=root_p,
                         save_data_path=save_data_path)
    else:
        print('\n'.join(['you can try: ',
                         '1. python exp_sr_test01.py run_test 50',
                         '2. python exp_sr_test01.py show_test',
                         '3. python exp_sr_test01.py gen_figures']))


if __name__ == '__main__':
    main()
