# -*- coding: utf-8 -*-
"""
In this test, we compare GraphStoIHT with three baseline methods
on the benchmark dataset, which can be found in reference [1].
References:
    [1] Arias-Castro, Ery, Emmanuel J. Candes, and Arnaud Durand.
    "Detection of an anomalous cluster in a network."
    The Annals of Statistics (2011): 278-304.
"""
import os
import time
import pickle
import random
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


bench_data = {
    # figure 1 in [1], it has 26 nodes.
    'fig_1': [475, 505, 506, 507, 508, 509, 510, 511, 512, 539, 540, 541, 542,
              543, 544, 545, 576, 609, 642, 643, 644, 645, 646, 647, 679, 712],
    # figure 2 in [1], it has 46 nodes.
    'fig_2': [439, 440, 471, 472, 473, 474, 504, 505, 506, 537, 538, 539, 568,
              569, 570, 571, 572, 600, 601, 602, 603, 604, 605, 633, 634, 635,
              636, 637, 666, 667, 668, 698, 699, 700, 701, 730, 731, 732, 733,
              763, 764, 765, 796, 797, 798, 830],
    # figure 3 in [1], it has 92 nodes.
    'fig_3': [151, 183, 184, 185, 217, 218, 219, 251, 252, 285, 286, 319, 320,
              352, 353, 385, 386, 405, 406, 407, 408, 409, 419, 420, 437, 438,
              439, 440, 441, 442, 443, 452, 453, 470, 471, 475, 476, 485, 486,
              502, 503, 504, 507, 508, 509, 518, 519, 535, 536, 541, 550, 551,
              568, 569, 583, 584, 601, 602, 615, 616, 617, 635, 636, 648, 649,
              668, 669, 670, 680, 681, 702, 703, 704, 711, 712, 713, 736, 737,
              738, 739, 740, 741, 742, 743, 744, 745, 771, 772, 773, 774, 775,
              776],
    # figure 4 in [1], it has 132 nodes.
    'fig_4': [244, 245, 246, 247, 248, 249, 254, 255, 256, 277, 278, 279, 280,
              281, 282, 283, 286, 287, 288, 289, 290, 310, 311, 312, 313, 314,
              315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 343, 344, 345,
              346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 377,
              378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390,
              411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423,
              448, 449, 450, 451, 452, 453, 454, 455, 456, 481, 482, 483, 484,
              485, 486, 487, 488, 489, 514, 515, 516, 517, 518, 519, 520, 521,
              547, 548, 549, 550, 551, 552, 553, 579, 580, 581, 582, 583, 584,
              585, 586, 613, 614, 615, 616, 617, 618, 646, 647, 648, 649, 650,
              680, 681],
    # grid size (length).
    'height': 33,
    # grid width (width).
    'width': 33,
    # the dimension of grid graph is 33 x 33.
    'p': 33 * 33,
    # sparsity list of these 4 figures.
    's': {'fig_1': 26, 'fig_2': 46, 'fig_3': 92, 'fig_4': 132},
    # sparsity list
    's_list': [26, 46, 92, 132],
    'graph': simu_grid_graph(height=33, width=33)
}


def algo_iht(x_mat, y_tr, max_epochs, lr, s, x0, tol_algo):
    """ Iterative Hard Thresholding Method proposed in reference [3]. The
        standard iterative hard thresholding method for compressive sensing.
    :param x_mat:       the design matrix.
    :param y_tr:        the array of measurements.
    :param max_epochs:  the maximum epochs (iterations) allowed.
    :param lr:          the learning rate (should be 1.0).
    :param s:           the sparsity parameter.
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
    run_time = time.time() - start_time
    return num_epochs, run_time, x_hat


def cv_iht(x_tr_mat, y_tr, x_va_mat, y_va, max_epochs, lr_list, s, x_star,
           x0, tol_algo):
    """ Tuning parameter by using additional validation dataset. """
    test_err_mat = np.zeros(shape=len(lr_list))
    x_hat_dict = dict()
    for lr_ind, lr in enumerate(lr_list):
        num_epochs, run_time, x_hat = algo_iht(
            x_mat=x_tr_mat, y_tr=y_tr, max_epochs=max_epochs,
            lr=lr, s=s, x0=x0, tol_algo=tol_algo)
        y_err = np.linalg.norm(y_va - np.dot(x_va_mat, x_hat)) ** 2.
        test_err_mat[lr_ind] = y_err
        x_hat_dict[lr] = (num_epochs, run_time, x_hat)
    min_index = np.argmin(test_err_mat)
    best_lr = lr_list[min_index]
    err = np.linalg.norm(x_star - x_hat_dict[best_lr][2])
    num_epochs, run_time = x_hat_dict[best_lr][:2]
    return err, num_epochs, run_time


def algo_sto_iht(x_mat, y_tr, max_epochs, lr, s, x0, tol_algo, b):
    """ Stochastic Iterative Hard Thresholding Method proposed in [1].
    :param x_mat:       the design matrix.
    :param y_tr:        the array of measurements.
    :param max_epochs:  the maximum epochs (iterations) allowed.
    :param lr:          the learning rate (should be 1.0).
    :param s:           the sparsity parameter.
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
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return num_epochs, run_time, x_hat


def cv_sto_iht(x_tr_mat, y_tr, x_va_mat, y_va, max_epochs, s, x_star, x0,
               tol_algo, b_list, lr_list):
    """ Tuning parameter by using additional validation dataset. """
    test_err_mat = np.zeros(len(lr_list) * len(b_list))
    para_dict = dict()
    x_hat_dict = dict()
    for index, (lr, b) in enumerate(product(lr_list, b_list)):
        num_epochs, run_time, x_hat = algo_sto_iht(
            x_mat=x_tr_mat, y_tr=y_tr, max_epochs=max_epochs,
            lr=lr, s=s, x0=x0, tol_algo=tol_algo, b=b)
        y_err = np.linalg.norm(y_va - np.dot(x_va_mat, x_hat)) ** 2.
        test_err_mat[index] = y_err
        para_dict[index] = (lr, b)
        x_hat_dict[(lr, b)] = (num_epochs, run_time, x_hat)
    lr, b = para_dict[int(np.argmin(test_err_mat))]
    err = np.linalg.norm(x_star - x_hat_dict[(lr, b)][2])
    num_epochs, run_time = x_hat_dict[(lr, b)][:2]
    return err, num_epochs, run_time


def algo_graph_iht(
        x_mat, y_tr, max_epochs, lr, x0, tol_algo, edges, costs, s,
        g=1, root=-1, gamma=0.1, proj_max_num_iter=50, verbose=0):
    """ Graph Iterative Hard Thresholding proposed in [4] and projection
            operator is proposed in [2].
    :param x_mat:       the design matrix.
    :param y_tr:        the array of measurements.
    :param max_epochs:  the maximum epochs (iterations) allowed.
    :param lr:          the learning rate (should be 1.0).
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
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return num_epochs, run_time, x_hat


def cv_graph_iht(x_tr_mat, y_tr, x_va_mat, y_va, max_epochs, lr_list, x_star,
                 x0, tol_algo, edges, costs, s):
    """ Tuning parameter by using additional validation dataset. """
    test_err_mat = np.zeros(len(lr_list))
    x_hat_dict = dict()
    for lr_ind, lr in enumerate(lr_list):
        num_epochs, run_time, x_hat = algo_graph_iht(
            x_mat=x_tr_mat, y_tr=y_tr, max_epochs=max_epochs,
            lr=lr, x0=x0, tol_algo=tol_algo, edges=edges, costs=costs, s=s)
        y_err = np.linalg.norm(y_va - np.dot(x_va_mat, x_hat)) ** 2.
        test_err_mat[lr_ind] = y_err
        x_hat_dict[lr] = (num_epochs, run_time, x_hat)
    min_index = np.argmin(test_err_mat)
    best_lr = lr_list[min_index]
    err = np.linalg.norm(x_star - x_hat_dict[best_lr][2])
    num_epochs, run_time = x_hat_dict[best_lr][:2]
    return err, num_epochs, run_time


def algo_graph_sto_iht(
        x_mat, y_tr, max_epochs, lr, x0, tol_algo, edges, costs, s, b,
        g=1, root=-1, gamma=0.1, proj_max_num_iter=50, verbose=0):
    """ Graph Stochastic Iterative Hard Thresholding.
    :param x_mat:       the design matrix.
    :param y_tr:        the array of measurements.
    :param max_epochs:  the maximum epochs (iterations) allowed.
    :param lr:          the learning rate (should be 1.0).
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
    run_time = time.time() - start_time
    return num_epochs, run_time, x_hat


def cv_graph_sto_iht(x_tr_mat, y_tr, x_va_mat, y_va, b_list, lr_list,
                     max_epochs, tol_algo, x_star, x0, edges, costs, s):
    """ Tuning parameter by using additional validation dataset. """
    test_err_mat = np.zeros(len(lr_list) * len(b_list))
    para_dict = dict()
    x_hat_dict = dict()
    for index, (lr, b) in enumerate(product(lr_list, b_list)):
        num_epochs, run_time, x_hat = algo_graph_sto_iht(
            x_mat=x_tr_mat, y_tr=y_tr, max_epochs=max_epochs,
            lr=lr, x0=x0, tol_algo=tol_algo, edges=edges,
            costs=costs, s=s, b=b)
        y_err = np.linalg.norm(y_va - np.dot(x_va_mat, x_hat)) ** 2.
        test_err_mat[index] = y_err
        para_dict[index] = (lr, b)
        x_hat_dict[(lr, b)] = (num_epochs, run_time, x_hat)
    lr, b = para_dict[int(np.argmin(test_err_mat))]
    err = np.linalg.norm(x_star - x_hat_dict[(lr, b)][2])
    num_epochs, run_time = x_hat_dict[(lr, b)][:2]
    return err, num_epochs, run_time


def print_helper(method, trial_i, n, err, num_epochs, run_time):
    print('%-13s trial_%03d n: %03d w_error: %.3e '
          'num_epochs: %03d run_time: %.3e' %
          (method, trial_i, n, err, num_epochs, run_time))


def run_single_test(data):
    s, n, p, x_star = data['s'], data['n'], data['p'], data['x_star']
    tol_algo, tol_rec = data['tol_algo'], data['tol_rec']
    max_epochs, fig_i = data['max_epochs'], data['fig_i']
    b_list, lr_list = data['b_list'], data['lr_list']
    trial_i = data['trial_i']
    x0 = np.zeros_like(x_star)
    x_tr_mat, y_tr = data['x_tr_mat'], data['y_tr']
    x_va_mat, y_va = data['x_va_mat'], data['y_va']
    edges = data['proj_para']['edges']
    costs = data['proj_para']['costs']

    rec_err = []
    # ------------ IHT ------------
    err, num_epochs, run_time = cv_iht(
        x_tr_mat=x_tr_mat, y_tr=y_tr, x_va_mat=x_va_mat, y_va=y_va,
        max_epochs=max_epochs, lr_list=lr_list, s=s, x_star=x_star, x0=x0,
        tol_algo=tol_algo)
    rec_err.append(('iht', err))
    print_helper('iht', trial_i, n, err, num_epochs, run_time)

    # ------------ StoIHT ------------
    err, num_epochs, run_time = cv_sto_iht(
        x_tr_mat=x_tr_mat, y_tr=y_tr, x_va_mat=x_va_mat, y_va=y_va,
        b_list=b_list, lr_list=lr_list, s=s, max_epochs=max_epochs,
        x_star=x_star, x0=x0, tol_algo=tol_algo)
    rec_err.append(('sto-iht', err))
    print_helper('sto-iht', trial_i, n, err, num_epochs, run_time)

    # ------------ GraphIHT ------------
    err, num_epochs, run_time = cv_graph_iht(
        x_tr_mat=x_tr_mat, y_tr=y_tr, x_va_mat=x_va_mat, y_va=y_va,
        max_epochs=max_epochs, lr_list=lr_list, x_star=x_star, x0=x0,
        tol_algo=tol_algo, edges=edges, costs=costs, s=s)
    rec_err.append(('graph-iht', err))
    print_helper('graph-iht', trial_i, n, err, num_epochs, run_time)

    # ------------ GraphStoIHT ------------
    err, num_epochs, run_time = cv_graph_sto_iht(
        x_tr_mat=x_tr_mat, y_tr=y_tr, x_va_mat=x_va_mat, y_va=y_va,
        b_list=b_list, lr_list=lr_list, max_epochs=max_epochs,
        tol_algo=tol_algo, x_star=x_star, x0=x0, edges=edges, costs=costs, s=s)
    rec_err.append(('graph-sto-iht', err))
    print_helper('graph-sto-iht', trial_i, n, err, num_epochs, run_time)
    return fig_i, trial_i, n, rec_err


def run_test(trial_range, fig_list, n_range_list, num_va, lr_list,
             tol_algo, tol_rec, max_epochs, num_cpus, root_p):
    np.random.seed()
    start_time = time.time()
    input_data_list = []
    for trial_i in trial_range:
        for fig_i, n_list in zip(fig_list, n_range_list):
            p, s = bench_data['p'], bench_data['s'][fig_i]
            para = {
                'trial_i': trial_i,
                's': s,
                'fig_i': fig_i,
                'p': p,
                'max_epochs': max_epochs,
                'n_list': n_list,
                'bench_data': bench_data,
                'tol_rec': tol_rec,
                'tol_algo': tol_algo,
                'verbose': 0,
                'proj_para': {'edges': bench_data['graph'][0],
                              'costs': bench_data['graph'][1]}
            }
            for n in n_list:
                x_tr_mat = np.random.normal(size=n * p)
                x_va_mat = np.random.normal(size=num_va * p)
                x_star = np.zeros(p)  # using Gaussian signal.
                x_star[bench_data[fig_i]] = np.random.normal(size=s)
                x_tr_mat = np.reshape(x_tr_mat, (n, p)) / np.sqrt(n)
                x_va_mat = np.reshape(x_va_mat, (num_va, p)) / np.sqrt(num_va)
                y_tr = np.dot(x_tr_mat, x_star)
                y_va = np.dot(x_va_mat, x_star)
                data = para.copy()
                data.update({
                    'n': n,
                    'x_tr_mat': x_tr_mat,
                    'x_va_mat': x_va_mat,
                    'x_star': x_star,
                    'b_list': [int(n) / 5, int(n) / 10],
                    'lr_list': lr_list,
                    'y_tr': y_tr,
                    'y_va': y_va})
                input_data_list.append(data)
    pool = multiprocessing.Pool(processes=num_cpus)
    results_pool = pool.map(run_single_test, input_data_list)
    pool.close()
    pool.join()

    sum_results = dict()
    for fig_i, trial_i, n, rec_err in results_pool:
        if trial_i not in sum_results:
            sum_results[trial_i] = []
        sum_results[trial_i].append((fig_i, trial_i, n, rec_err))
    for trial_i in sum_results:
        f_name = root_p + 'results_exp_sr_test04_trial_%02d.pkl' % trial_i
        print('save results to file: %s' % f_name)
        pickle.dump({'results_pool': sum_results[trial_i]},
                    open(f_name, 'wb'))
    print('total run time of %02d trials: %.2f seconds.' %
          (len(trial_range), time.time() - start_time))


def summarize_results(
        trial_list, fig_list, n_range_list, method_list, tol_rec, root_p,
        trim_ratio):
    results_pool = []
    num_trials = len(trial_list)
    for trial_i in trial_list:
        f_name = root_p + 'results_exp_sr_test04_trial_%02d.pkl' % trial_i
        results = pickle.load(open(f_name))
        for item in results:
            results_pool.extend(results[item])
    sum_results = dict()
    for fig_i, trial_i, n, rec_err in results_pool:
        fig_i_ind = list(fig_list).index(fig_i)
        n_ind = list(n_range_list[fig_i_ind]).index(n)
        trial_i_ind = list(trial_list).index(trial_i)
        for method, err in rec_err:
            if fig_i not in sum_results:
                sum_results[fig_i] = dict()
            if method not in sum_results[fig_i]:
                sum_results[fig_i][method] = np.zeros(
                    (num_trials, len(n_range_list[fig_i_ind])))
            sum_results[fig_i][method][trial_i_ind][n_ind] = err
    # trim 5% of the results (rounding when necessary).
    num_trim = int(round(trim_ratio * num_trials))
    trim_results = {
        fig_i: {
            method: np.zeros(
                (num_trials - 2 * num_trim, len(n_range_list[ind])))
            for method in method_list}
        for ind, fig_i in enumerate(sum_results)}
    for fig_i in sum_results:
        for method in sum_results[fig_i]:
            # remove 5% best and 5% worst.
            re = sum_results[fig_i][method]
            trimmed_re = np.sort(re, axis=0)[num_trim:num_trials - num_trim, :]
            trim_results[fig_i][method] = trimmed_re
    for fig_i in trim_results:
        for method in trim_results[fig_i]:
            re = trim_results[fig_i][method]
            re[re > tol_rec] = 0.
            re[re != 0.0] = 1.0
            trim_results[fig_i][method] = np.mean(re, axis=0)
    f_name = root_p + 'results_exp_sr_test04.pkl'
    print('save file to: %s' % f_name)
    pickle.dump({'trim_results': trim_results,
                 'sum_results': sum_results,
                 'results_pool': results_pool},
                open(f_name, 'wb'))


def show_test(n_range_list, method_list, title_list, fig_list, root_p):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 16, 8
    color_list = ['b', 'g', 'm', 'r']
    marker_list = ['X', 'o', 'P', 's']
    x_label = n_range_list
    s_list = [26, 46, 92, 132]
    f_name = 'results_exp_sr_test04.pkl'
    trim_results = pickle.load(open(root_p + f_name))['trim_results']

    fig, ax = plt.subplots(2, 4)
    for fig_ind, fig_i in enumerate(fig_list):
        height, width = bench_data['height'], bench_data['width']
        x_star = np.ones((height, width)) * np.nan
        for node_id in bench_data[fig_i]:
            i, j = node_id / height, node_id % width
            x_star[i, j] = np.random.normal(0.0, 1.0)
        for i in range(height + 1):
            ax[0, fig_ind].axhline(i, lw=.01, color='gray',
                                   linestyle='dotted', zorder=5)
        for i in range(width + 1):
            ax[0, fig_ind].axvline(i, lw=.01, color='gray',
                                   linestyle='dotted', zorder=5)
        ax[0, fig_ind].imshow(x_star, interpolation='none', cmap='binary',
                              extent=[0, height, 0, width], zorder=0)
        ax[0, fig_ind].axes.get_xaxis().set_visible(False)
        ax[0, fig_ind].axes.get_yaxis().set_visible(False)
    for ii in range(4):
        ax[1, ii].grid(b=True, which='both', color='lightgray',
                       linestyle='dotted', axis='both')
    ax[1, 0].set_xticks([2.5, 4.0, 5.5, 7.0])
    ax[1, 1].set_xticks([2.5, 3.5, 4.5])
    ax[1, 2].set_xticks([2.0, 2.5, 3.0, 3.5, 4.0])
    ax[1, 3].set_xticks([2.0, 2.5, 3.0, 3.5, 4.0])

    for fig_ind, fig_i in enumerate(fig_list):
        for method_ind, method in enumerate(method_list):
            ax[1, fig_ind].plot(
                np.asarray(x_label[fig_ind]) / (1. * s_list[fig_ind]),
                trim_results[fig_i][method],
                c=color_list[method_ind], linestyle='-',
                markerfacecolor='none',
                marker=marker_list[method_ind], markersize=8.,
                markeredgewidth=1.5, linewidth=2.0,
                label=title_list[method_ind])
    ax[1, 0].set_ylabel(r'Probability of Recovery')
    for i in range(4):
        ax[1, i].set_xlabel(r'Oversampling ratio $\displaystyle m / s$')
        ax[1, i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1, 1].legend(loc='lower right', fontsize=18.,
                    bbox_to_anchor=(0.25, 0.05), borderpad=0.1,
                    labelspacing=0.2, handletextpad=0.1)
    title_list = ['(a) Graph-01', '(b) Graph-02',
                  '(c) Graph-03', '(d) Graph-04']
    for i in range(4):
        ax[0, i].set_title(title_list[i])
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
    plt.setp(ax[1, 1].get_yticklabels(), visible=False)
    plt.setp(ax[1, 2].get_yticklabels(), visible=False)
    plt.setp(ax[1, 3].get_yticklabels(), visible=False)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    f_name = root_p + 'results_exp_sr_test04.pdf'
    print('save fig to: %s' % f_name)
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0,
                format='pdf')
    plt.close()


def main():
    # try 50 different trials and take average on 44 trials.
    num_trials = 50
    # maximum number of epochs
    max_epochs = 500
    # tolerance of the algorithm
    tol_algo = 1e-7
    # tolerance of the recovery.
    tol_rec = 1e-6
    # the trimmed ratio
    # ( about 5% of the best and worst have been removed).
    trim_ratio = 0.05
    # sparsity considered.
    num_va = 100
    # learning rate
    lr_list = [0.2, 0.4, 0.6, 0.8]

    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    method_list = ['iht', 'sto-iht', 'graph-iht', 'graph-sto-iht']
    title_list = ['IHT', 'StoIHT', 'GraphIHT', 'GraphStoIHT']
    n_range_list = [range(30, 201, 10), range(70, 250, 10),
                    range(170, 400, 10), range(250, 500, 10)]

    # TODO config the path by yourself.
    root_p = 'results/'
    if not os.path.exists(root_p):
        os.mkdir(root_p)

    if len(os.sys.argv) <= 1:
        print('\n'.join(['please use one of the following commands: ',
                         '1. python exp_sr_test04.py run_test',
                         '2. python exp_sr_test04.py show_test']))
        exit(0)

    command = os.sys.argv[1]
    if command == 'run_test':
        num_cpus = int(os.sys.argv[2])
        trial_range = range(int(os.sys.argv[3]), int(os.sys.argv[4]))
        for trial_i in trial_range:
            run_test(trial_range=[trial_i],
                     fig_list=fig_list,
                     n_range_list=n_range_list,
                     tol_algo=tol_algo,
                     num_va=num_va,
                     lr_list=lr_list,
                     tol_rec=tol_rec,
                     max_epochs=max_epochs,
                     num_cpus=num_cpus,
                     root_p=root_p)
    elif command == 'summarize_results':
        trial_range = range(num_trials)
        summarize_results(trial_list=trial_range,
                          fig_list=fig_list,
                          n_range_list=n_range_list,
                          method_list=method_list,
                          tol_rec=tol_rec,
                          root_p=root_p,
                          trim_ratio=trim_ratio)
    elif command == 'show_test':
        show_test(n_range_list=n_range_list,
                  method_list=method_list,
                  title_list=title_list,
                  fig_list=fig_list,
                  root_p=root_p)


if __name__ == '__main__':
    main()
