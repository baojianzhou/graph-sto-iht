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
    np.random.seed()
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
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
    x_err = np.linalg.norm(x_hat - x_star)
    run_time = time.time() - start_time
    return x_err, num_epochs, run_time


def print_helper(method, trial_i, b, n, num_epochs, err, run_time):
    print('%13s   trial_%03d b: %02d n: %03d num_epochs: %03d '
          'rec_error: %.6e run_time: %.6e' %
          (method, trial_i, b, n, num_epochs, err, run_time))


def run_single_test(data):
    np.random.seed()
    # generate Gaussian measurement matrix.
    # each entry is generated from N(0,1)/sqrt(n) Gaussian.
    s, n, p, b = data['s'], data['n'], data['p'], data['b']
    lr = data['lr']
    x0 = data['x0']
    x_star = data['x_star']
    x_mat = data['x_mat']
    y_list = data['y_list']
    trial_i = data['trial_i']
    tol_algo = data['tol_algo']
    max_epochs = data['max_epochs']

    edges = data['proj_para']['edges']
    costs = data['proj_para']['costs']

    rec_err = []
    for noise_level, y_tr in y_list:
        if b == 2:  # just need to run once.
            err, num_epochs, run_time = algo_graph_iht(
                x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs, lr=lr,
                x_star=x_star, x0=x0, tol_algo=tol_algo, edges=edges,
                costs=costs, s=s)
            rec_err.append(('graph-iht', b, noise_level, err))
            print_helper('graph-iht', trial_i, b, n, num_epochs, err, run_time)
        err, num_epochs, run_time = algo_graph_sto_iht(
            x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs, lr=lr,
            x_star=x_star, x0=x0, tol_algo=tol_algo, edges=edges, costs=costs,
            s=s, b=b)
        rec_err.append(('graph-sto-iht', b, noise_level, err))
        print_helper('graph-sto-iht', trial_i, b, n, num_epochs, err, run_time)
    return trial_i, b, n, rec_err


def run_test(s, n_list, p, lr, height, width, max_epochs, tol_algo,
             tol_rec, b_list, num_cpus, trial_range, root_p, noise_level_list):
    # make sure, it works under multiprocessing case.
    np.random.seed()
    start_time = time.time()
    # size of grid graph
    for trial_i in trial_range:
        input_data_list, saved_data = [], dict()
        x_mat = np.random.normal(loc=0.0, scale=1.0, size=300 * p)
        edges, costs = simu_grid_graph(height=height, width=width)
        init_node = (height / 2) * width + height / 2
        sub_graphs = {_: random_walk(edges=edges, s=_, init_node=init_node)
                      for _ in [s]}
        x_star = np.zeros(p)  # using Gaussian signal.
        x_star[sub_graphs[s][0]] = np.random.normal(loc=0.0, scale=1.0, size=s)
        for (b, n) in product(b_list, n_list):
            # set block size
            print('generate_data pair: (trial_%03d, b: %02d, n: %03d)' %
                  (trial_i, b, n))
            x_n_mat = np.reshape(x_mat[:n * p], (n, p)) / np.sqrt(n)
            y_tr = np.dot(x_n_mat, x_star)
            y_list = []
            for noise_level in noise_level_list:
                noise_e = np.random.normal(loc=0.0, scale=1.0, size=len(y_tr))
                noise_e = (noise_level / np.linalg.norm(noise_e)) * noise_e
                y_list.append([noise_level, y_tr + noise_e])
            data = {'lr': lr,
                    'max_epochs': max_epochs,
                    'trial_i': trial_i,
                    's': s,
                    'n': n,
                    'x_mat': x_n_mat,
                    'y_list': y_list,
                    'n_list': n_list,
                    'b_list': b_list,
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
        results_pool = pool.map(run_single_test, input_data_list)
        pool.close()
        pool.join()
        f_name = root_p + 'sr_simu_test03_trial_%02d.pkl' % trial_i
        print('save results to file: %s' % f_name)
        pickle.dump({'results_pool': results_pool},
                    open(f_name, 'wb'))
        print('total run time of %02d trials: %.2f seconds.' %
              (len(trial_range), time.time() - start_time))


def summarize_results(
        trim_ratio, trial_range, n_list, b_list, method_list,
        root_p, noise_level_list):
    results_pool = []
    num_trials = len(trial_range)
    for trial_i in trial_range:
        f_name = root_p + 'sr_simu_test03_trial_%02d.pkl' % trial_i
        print('load file: %s' % f_name)
        for item in pickle.load(open(f_name))['results_pool']:
            results_pool.append(item)
    # try to trim 5% of the results (rounding when necessary).
    num_trim = int(round(trim_ratio * num_trials))
    sum_results = dict()
    trim_results = dict()
    for noise_level in noise_level_list:
        sum_results[noise_level] = dict()
        trim_results[noise_level] = dict()
        for method in method_list:
            sum_results[noise_level][method] = dict()
            trim_results[noise_level][method] = dict()
            for b in b_list:
                re = np.zeros((num_trials, len(n_list)))
                sum_results[noise_level][method][b] = re
                re = np.zeros((num_trials - 2 * num_trim, len(n_list)))
                trim_results[noise_level][method][b] = re
    # trial_i, b, n, rec_err
    for trial_i, b, n, re in results_pool:
        n_ind = list(n_list).index(n)
        for method, _, noise_level, val in re:
            ind = list(trial_range).index(trial_i)
            if method in method_list and noise_level in noise_level_list:
                sum_results[noise_level][method][b][ind][n_ind] = val
    for flag in noise_level_list:
        for method, b in product(['graph-iht'], b_list):
            if b != 2:
                sum_results[flag][method][b] = sum_results[flag][method][2]
    for noise_level, threshold in zip(noise_level_list,
                                      [1e-6, 0.3, 0.5, 0.7]):
        for method in sum_results[noise_level]:
            for b in sum_results[noise_level][method]:
                re = sum_results[noise_level][method][b]
                start = num_trim
                end = num_trials - num_trim
                # remove 5% best and 5% worst.
                trimmed_re = np.sort(re, axis=0)[start:end, :]
                trimmed_re[trimmed_re > threshold] = 0.0
                trimmed_re[trimmed_re != 0.0] = 1.0
                re = np.mean(trimmed_re, axis=0)
                feasible_indices = np.where(re == 1.)[0]
                if len(feasible_indices) == 0:  # set to largest.
                    least_n = n_list[-1]
                else:
                    least_n = n_list[feasible_indices[0]]
                trim_results[noise_level][method][b] = least_n
    f_name = root_p + 'sr_simu_test03.pkl'
    print('save results to file: %s' % f_name)
    pickle.dump({'trim_results': trim_results,
                 'sum_results': sum_results,
                 'results_pool': results_pool}, open(f_name, 'wb'))


def show_test(b_list, method_list, title_list, root_p, noise_level_list):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 8, 4
    all_results = pickle.load(open(root_p + 'sr_simu_test03.pkl'))
    results = all_results['trim_results']
    color_list = ['b', 'g', 'm', 'r']
    marker_list = ['X', 'o', 'P', 's']
    fig, ax = plt.subplots(1, 2, sharey='all')
    for i in range(2):
        ax[i].grid(b=True, which='both', color='gray', linestyle='dotted',
                   axis='both')
        ax[i].set_xlim([0, 35])
        ax[i].set_ylim([0, 75])
    ax[0].set_xticks([4, 14, 24, 34])
    ax[0].set_yticks([0, 25, 50, 75])
    ax[1].set_xticks([4, 14, 24, 34, 44, 54, 64])
    ax[0].set_title(r"$\displaystyle \|{\bf \epsilon}\|=0.0$")
    ax[1].set_title(r"$\displaystyle \|{\bf \epsilon}\|=0.5$")
    ax[0].set_ylabel('Number of Measurements Required')
    x_list = b_list
    for method_ind, method in enumerate(method_list):
        for ind, noise_level in enumerate(noise_level_list):
            ax[0].plot(x_list, [results[noise_level][method][b]
                                for b in b_list],
                       label=str(noise_level),
                       color=color_list[ind],
                       marker=marker_list[ind]
                       , markersize=4.0, markerfacecolor='none',
                       linestyle='-', markeredgewidth=1.0, linewidth=1.0)
    for i in range(2):
        ax[i].set_xlabel('Block Size')
    ax[1].legend(loc='lower right', fontsize=14., borderpad=0.01,
                 labelspacing=0.0, handletextpad=0.05, framealpha=1.0)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    f_name = root_p + 'sr_simu_test03.pdf'
    print('save fig to: %s' % f_name)
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0,
                format='pdf')


def main():
    # try 50 different trials and take average on 44 trials.
    num_trials = 50
    # maximum number of epochs
    max_epochs = 500
    # tolerance of the algorithm
    tol_algo = 1e-7
    # tolerance of the recovery.
    tol_rec = 1e-6
    # the dimension of the grid graph.
    p = 256
    # the trimmed ratio
    # ( about 5% of the best and worst have been removed).
    trim_ratio = 0.05
    # height and width of the grid graph.
    height, width = 16, 16
    # different block size parameters considered.
    b_list = range(2, 65, 2)
    # number of measurements list
    n_list = range(20, 201, 5)
    # sparsity considered.
    s = 8
    # learning rate
    lr = 1.0
    # list of methods
    method_list = ['graph-iht', 'graph-sto-iht']
    # noise level list
    noise_level_list = [0.0, 0.3, 0.5, 0.7]
    # TODO config by yourself.
    root_p = '/network/rit/lab/ceashpc/bz383376/data/icml19/publish/'
    if not os.path.exists(root_p):
        os.mkdir(root_p)
    command = os.sys.argv[1]
    if command == 'run_test':
        num_cpus = int(os.sys.argv[2])
        trial_range = range(int(os.sys.argv[3]), int(os.sys.argv[4]))
        run_test(s=s,
                 n_list=n_list,
                 p=p,
                 lr=lr,
                 height=height,
                 width=width,
                 max_epochs=max_epochs,
                 tol_algo=tol_algo,
                 tol_rec=tol_rec,
                 b_list=b_list,
                 num_cpus=num_cpus,
                 trial_range=trial_range,
                 root_p=root_p,
                 noise_level_list=noise_level_list)
    elif command == 'summarize_results':
        trial_range = range(num_trials)
        summarize_results(trim_ratio=trim_ratio,
                          trial_range=trial_range,
                          b_list=b_list,
                          n_list=n_list,
                          method_list=method_list,
                          root_p=root_p,
                          noise_level_list=noise_level_list)
    elif command == 'show_test':
        title_list = ['IHT', 'StoIHT', 'GraphIHT', 'GraphStoIHT']
        trial_range = range(50, 52)
        summarize_results(trim_ratio=trim_ratio,
                          trial_range=trial_range,
                          b_list=b_list,
                          n_list=n_list,
                          method_list=method_list,
                          root_p=root_p,
                          noise_level_list=noise_level_list)
        show_test(b_list=b_list,
                  method_list=method_list,
                  title_list=title_list,
                  root_p=root_p,
                  noise_level_list=noise_level_list)


if __name__ == '__main__':
    main()
