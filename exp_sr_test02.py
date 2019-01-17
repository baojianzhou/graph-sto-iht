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
from os import sys
import multiprocessing
from itertools import product
import numpy as np

try:
    import sparse_module

    try:
        from sparse_module import wrap_head_tail_binsearch
    except ImportError:
        print('cannot find wrap_head_tail_binsearch method in sparse_module')
        sparse_module = None
        exit(0)
except ImportError:
    print('\n'.join(['cannot find the module: sparse_module',
                     'use ./build.sh build sparse_module.so library.']))


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
    re_nodes = wrap_head_tail_binsearch(
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

    x_err_list = []
    x_iter_err_list = []
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
            x_iter_err_list.append(np.linalg.norm(x_hat - x_star))

        x_err_list.append(np.linalg.norm(x_hat - x_star))
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
    x_err = np.linalg.norm(x_hat - x_star)
    run_time = time.time() - start_time
    return x_err_list, x_iter_err_list, x_err, num_epochs, run_time


def print_helper(method, trial_i, b, n, num_epochs, err, run_time):
    print('%13s trial_%03d b: %03d n: %03d num_epochs: %03d '
          'rec_error: %.3e run_time: %.3e' %
          (method, trial_i, b, n, num_epochs, err, run_time))


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

    rec_error = []
    # ------------- GraphStoIHT --------
    x_err_list, _, err, num_epochs, run_time = algo_graph_sto_iht(
        x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs, lr=lr, x_star=x_star,
        x0=x0, tol_algo=tol_algo, edges=edges, costs=costs, s=s, b=b)
    rec_error.append(('graph-sto-iht', x_err_list))
    print_helper('graph-sto-iht', trial_i, b, n, num_epochs, err, run_time)
    return trial_i, b, rec_error


def run_single_test_diff_eta(data):
    np.random.seed()
    # generate Gaussian measurement matrix.
    # each entry is generated from N(0,1)/sqrt(n) Gaussian.
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
    # ------------- GraphStoIHT --------
    _, x_err_iter_list, err, num_epochs, run_time = algo_graph_sto_iht(
        x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs, lr=lr, x_star=x_star,
        x0=x0, tol_algo=tol_algo, edges=edges, costs=costs, s=s, b=b)
    rec_error.append(('graph-sto-iht', x_err_iter_list))
    print_helper('graph-sto-iht', trial_i, b, n, num_epochs, err, run_time)
    return trial_i, lr, rec_error


def run_test_diff_b(
        s, p, height, width, max_epochs, tol_algo, tol_rec, b_list,
        trim_ratio, num_cpus, num_trials):
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
            'lr': {b: 1.0 if b != 180 else 1.0 / 2. for b in b_list},
            'max_epochs': max_epochs,
            'trial_i': trial_i,
            's': s,
            'n': 180,
            'n_list': [180],
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
            'proj_para': {'edges': edges, 'costs': costs}
        }
        if s not in saved_data:
            saved_data[s] = data
        input_data_list.append(data)
    pool = multiprocessing.Pool(processes=num_cpus)
    results_pool = pool.map(run_single_test_diff_b, input_data_list)
    pool.close()
    pool.join()
    x_len = 30
    sum_results = {method: {trial_i: [None] * len(b_list)
                            for trial_i in range(num_trials)}
                   for method in ['graph-sto-iht']}
    # # try to trim 5% of the results (rounding when necessary).
    num_trim = int(trim_ratio * num_trials)
    for trial_i, b, re in results_pool:
        b_ind = list(b_list).index(b)
        for method, rec_err in re:
            sum_results[method][trial_i][b_ind] = rec_err
    trim_results = {method: dict() for method in ['graph-sto-iht']}
    for method in ['graph-sto-iht']:
        for b_ind in range(len(b_list)):
            average_re = np.zeros((num_trials, x_len))
            for trial_i in sum_results[method]:
                re = sum_results[method][trial_i][b_ind]
                if len(re) < x_len:
                    ext_re = np.zeros(x_len)
                    ext_re[:len(re)] = re
                    ext_re[len(re):] = [re[-1]] * (x_len - len(re))
                else:
                    ext_re = re[:x_len]
                average_re[trial_i] = ext_re
            # remove 5% best and 5% worst.
            trim_re = np.zeros((num_trials - 2 * num_trim, x_len))
            for i in range(x_len):
                re = average_re[:, i]
                trim_re[:, i] = np.sort(re)[num_trim:len(re) - num_trim]
            re = np.log10(np.mean(trim_re, axis=0))
            trim_results[method][b_ind] = re
    print('total run time of %02d trials: %.2f seconds.' %
          (num_trials, time.time() - start_time))
    return {'trim_results': trim_results,
            'sum_results': sum_results,
            'saved_data': saved_data}


def run_test_diff_eta(
        s, p, lr_list, height, width, num_iterations, tol_algo, tol_rec, b,
        trim_ratio, num_cpus, num_trials):
    # make sure, it works under multiprocessing case.
    np.random.seed()
    start_time = time.time()
    # size of grid graph
    input_data_list, saved_data = [], dict()
    for (lr, trial_i) in product(lr_list, range(num_trials)):
        edges, costs = simu_grid_graph(height=height, width=width)
        init_node = (height / 2) * width + height / 2
        sub_graphs = {_: random_walk(edges=edges, s=_, init_node=init_node)
                      for _ in [s]}
        x_star = np.zeros(p)  # using Gaussian signal.
        x_star[sub_graphs[s][0]] = np.random.normal(loc=0.0, scale=1.0, size=s)
        data = {
            'lr': lr,
            'lr_list': lr_list,
            'max_epochs': 210,
            'trial_i': trial_i,
            's': s,
            'n': 80,
            'n_list': [80],
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
            'proj_para': {'edges': edges, 'costs': costs}
        }
        # we just save one case for each sparsity.
        if lr not in saved_data:
            saved_data[lr] = data
        input_data_list.append(data)
    pool = multiprocessing.Pool(processes=num_cpus)
    results_pool = pool.map(run_single_test_diff_eta, input_data_list)
    pool.close()
    pool.join()
    sum_results = {method: {trial_i: [None] * len(lr_list)
                            for trial_i in range(num_trials)}
                   for method in ['graph-sto-iht']}
    num_trim = int(trim_ratio * num_trials)  # try to trim 5% of the results.
    for trial_i, lr, re in results_pool:
        lr_ind = list(lr_list).index(lr)
        for method, rec_err in re:
            sum_results[method][trial_i][lr_ind] = rec_err
    trim_results = {method: dict() for method in ['graph-sto-iht']}
    for method in ['graph-sto-iht']:
        for lr_ind in range(len(lr_list)):
            average_re = np.zeros((num_trials, num_iterations))
            for trial_i in sum_results[method]:
                re = sum_results[method][trial_i][lr_ind]
                if len(re) < num_iterations:
                    ext_re = np.zeros(num_iterations)
                    ext_re[:len(re)] = re
                    ext_re[len(re):] = [re[-1]] * (num_iterations - len(re))
                else:
                    ext_re = re[:num_iterations]
                average_re[trial_i] = ext_re
            trim_re = np.zeros(
                (num_trials - 2 * num_trim, num_iterations))
            for i in range(num_iterations):
                re = average_re[:, i]
                trim_re[:, i] = np.sort(re)[num_trim:len(re) - num_trim]
            re = np.log10(np.mean(trim_re, axis=0))
            trim_results[method][lr_ind] = re
    print('total run time of %02d trials: %.2f seconds.' %
          (num_trials, time.time() - start_time))
    return {'trim_results': trim_results,
            'sum_results': sum_results,
            'saved_data': saved_data}


def show_test(lr_list, b_list, root_p):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    rc('text', usetex=True)

    rcParams['figure.figsize'] = 8, 3
    fig, ax = plt.subplots(1, 2)

    color_list = ['darkblue', 'blue', 'royalblue', 'dodgerblue', 'aqua',
                  'mediumspringgreen', 'greenyellow', 'yellow', 'orange',
                  'darkorange', 'red', 'magenta']
    marker_list = ['+', 'o', '*', '.', 'x', 's', 'D', '^', 'v', '>', '<', '']
    results = pickle.load(open(root_p + 'sr_simu_test02.pkl'))
    results_01 = results['re_diff_b']['trim_results']
    ax[0].set_yticklabels([
        r"$\displaystyle 10^{0}$",
        r"$\displaystyle 10^{-2}$",
        r"$\displaystyle 10^{-4}$",
        r"$\displaystyle 10^{-6}$",
        r"$\displaystyle 10^{-8}$"])
    ax[0].set_ylabel(r"$\displaystyle \| {\bf x} - \hat{{\bf x}}\|$")
    ax[0].set_xticks([0, 5, 10, 15, 20, 25])
    ax[0].grid(b=True, which='both', color='gray',
               linestyle='dotted', axis='both')
    ax[1].grid(b=True, which='both', color='gray',
               linestyle='dotted', axis='both')
    ax[0].set_xlim([-1, 30])
    ax[0].set_ylim([-8., 2.])
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    ax[0].set_yticks(ticks=[2, 0, -2, -4, -6, -8])
    ax[0].set_xlabel('Epoch')
    box = ax[0].get_position()
    ax[0].set_position(
        [box.x0, box.y0, box.width * 0.8, box.height])
    legend_list = []
    for b in b_list:
        legend_list.append(r"$\displaystyle b=%d$" % b)
    for i in range(30):
        row = [str(i)]
        row.extend(['%.4f' % results_01['graph-sto-iht'][_][i]
                    for _ in range(len(b_list))])
        print(','.join(row))
    results_02 = results['re_diff_eta']['trim_results']
    print('-' * 80)
    for i in range(100):
        row = [str(i)]
        row.extend(['%.4f' % results_02['graph-sto-iht'][_][i]
                    for _ in range(len(lr_list))])
        print(','.join(row))
    for b_ind in range(len(b_list)):
        label_ = legend_list[b_ind]
        if b_ind < 11:
            linestyle_ = '-'
        else:
            linestyle_ = '--'
        ax[0].plot(results_01['graph-sto-iht'][b_ind], color=color_list[b_ind],
                   marker=marker_list[b_ind], markersize=5.,
                   markerfacecolor='none', linestyle=linestyle_,
                   markeredgewidth=1.5, linewidth=1.5, label=label_)
    ax[0].legend(loc='center left', bbox_to_anchor=(0.99, 0.5),
                 fontsize=13., borderpad=0.1, labelspacing=0.2,
                 handletextpad=0.1)

    ax[1].set_xlim([0, 1000])
    ax[1].set_ylim([-8.0, 2])
    ax[1].set_xticks([0, 300, 600, 900])
    ax[1].set_yticks([2, 0, -2, -4, -6, -8])
    ax[1].set_yticklabels([
        r"$\displaystyle 10^{0}$",
        r"$\displaystyle 10^{-2}$",
        r"$\displaystyle 10^{-4}$",
        r"$\displaystyle 10^{-6}$",
        r"$\displaystyle 10^{-8}$"])
    colors = plt.cm.jet(np.asarray(lr_list) / 1.6)
    for lr_ind, lr in enumerate(lr_list):
        ax[1].plot(results_02['graph-sto-iht'][lr_ind],
                   linewidth=2., color=colors[lr_ind],
                   label=r"$\displaystyle \eta=%.1f$" % lr)
    # obtain the handles and labels from the figure
    handles, labels = ax[1].get_legend_handles_labels()
    # copy the handles
    import copy
    handles = [copy.copy(ha) for ha in handles]
    [ha.set_linewidth(10) for ha in handles]
    ax[1].legend(loc='center left', bbox_to_anchor=(0.98, .5),
                 fontsize=11.5, borderpad=0.1, labelspacing=0.05,
                 handletextpad=0.05)
    ax[1].set_xlabel('Iteration')
    ax[0].set_title('GraphStoIHT')
    ax[1].set_title('GraphStoIHT')
    plt.subplots_adjust(wspace=0.7, hspace=0.5)
    f_name = root_p + 'sr_simu_test02.pdf'
    print('save fig to: %s' % f_name)
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0,
                format='pdf')
    plt.close()


def main():
    # try 50 different trials and take average on 44 trials.
    num_trials = 50
    # tolerance of the algorithm
    tol_algo = 1e-7
    # tolerance of the recovery.
    tol_rec = 1e-6
    # the dimension of the grid graph.
    p = 256
    # the trimmed ratio ( about 5% of the best and worst have been removed).
    trim_ratio = 0.05
    # height and width of the grid graph.
    height, width = 16, 16
    s = 8
    # different sparsity parameters considered.
    b_list = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 180]
    # different learning rate
    lr_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
               1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

    # TODO config the path by yourself.
    root_p = 'results/'
    if not os.path.exists(root_p):
        os.mkdir(root_p)
    save_data_path = root_p + 'results_exp_sr_test02.pkl'

    if len(os.sys.argv) <= 1:
        print('\n'.join(['please use one of the following commands: ',
                         '1. python exp_sr_test02.py run_test',
                         '2. python exp_sr_test02.py show_test']))
        exit(0)

    command = sys.argv[1]
    if command == 'run_test':
        # learning rate
        num_cpus = int(sys.argv[2])
        re_diff_b = run_test_diff_b(s=s,
                                    p=p,
                                    height=height,
                                    width=width,
                                    max_epochs=35,
                                    tol_algo=tol_algo,
                                    tol_rec=tol_rec,
                                    b_list=b_list,
                                    trim_ratio=trim_ratio,
                                    num_cpus=num_cpus,
                                    num_trials=num_trials)

        re_diff_eta = run_test_diff_eta(s=s,
                                        p=p,
                                        lr_list=lr_list,
                                        height=height,
                                        width=width,
                                        num_iterations=2000,
                                        tol_algo=tol_algo,
                                        tol_rec=tol_rec,
                                        b=s,
                                        trim_ratio=trim_ratio,
                                        num_cpus=num_cpus,
                                        num_trials=num_trials)
        pickle.dump({'re_diff_b': re_diff_b,
                     're_diff_eta': re_diff_eta},
                    open(save_data_path, 'wb'))
    elif command == 'show_test':
        show_test(lr_list=lr_list, b_list=b_list, root_p=root_p)
    else:
        print('\n'.join(['you can try: ',
                         '1. python exp_sr_test02.py run_test 50',
                         '2. python exp_sr_test02.py show_test']))


if __name__ == '__main__':
    main()
