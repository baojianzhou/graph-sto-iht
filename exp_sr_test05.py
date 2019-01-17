# -*- coding: utf-8 -*-
import os
import time
import pickle
import random
from os import sys
import multiprocessing

from itertools import product

# you need to install numpy
import numpy as np

try:
    import sparse_module

    try:
        from sparse_module import wrap_head_tail_binsearch
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        sparse_module = None
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')

# TODO config by yourself.
root_p = '/network/rit/lab/ceashpc/bz383376/data/icml19/publish_2/'
if not os.path.exists(root_p):
    os.mkdir(root_p)


def algo_head_tail_binsearch(
        edges, x, costs, g, root, s_low, s_high, max_num_iter, verbose):
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
    p = len(x)
    x_mat = np.random.normal(0.0, 1.0, size=(n * p)) / np.sqrt(n)
    x_mat = x_mat.reshape((n, p))
    y_tr = np.dot(x_mat, x)
    noise_e = np.random.normal(0.0, 1.0, len(y_tr))
    y_e = y_tr + (norm_noise / np.linalg.norm(noise_e)) * noise_e
    return x_mat, y_tr, y_e


def algo_iht(x_mat, y_tr, max_epochs, lr, s, x0, tol_algo):
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


def cv_iht(x_tr_mat, y_tr, x_va_mat, y_va,
           max_epochs, lr_list, s, x_star, x0, tol_algo):
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
    min_index = np.argmin(test_err_mat)
    lr, b = para_dict[min_index]
    err = np.linalg.norm(x_star - x_hat_dict[(lr, b)][2])
    num_epochs, run_time = x_hat_dict[(lr, b)][:2]
    return err, num_epochs, run_time


def algo_graph_iht(
        x_mat, y_tr, max_epochs, lr, x0, tol_algo, edges, costs, g,
        root, s, h_factor, gamma, proj_max_num_iter, verbose):
    start_time = time.time()
    x_hat, num_iter = x0, 0
    x_tr_t = np.transpose(x_mat)
    xtx = np.dot(x_tr_t, x_mat)
    xty = np.dot(x_tr_t, y_tr)

    # graph info
    h_low = int(h_factor * s)
    h_high = int(h_factor * s * (1. + gamma))
    t_low = int(s)
    t_high = int(s * (1. + gamma))

    num_epochs = 0
    for epoch_i in range(max_epochs):
        num_epochs += 1
        grad = -1. * (xty - np.dot(xtx, x_hat))
        head_nodes, proj_gradient = algo_head_tail_binsearch(
            edges, grad, costs, g, root, h_low, h_high,
            proj_max_num_iter, verbose)
        bt = x_hat - lr * proj_gradient
        tail_nodes, proj_bt = algo_head_tail_binsearch(
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
                 x0, tol_algo, edges, costs, g, root, s, h_factor, gamma,
                 proj_max_num_iter, verbose):
    """ Tuning parameter by using additional validation dataset. """
    test_err_mat = np.zeros(len(lr_list))
    x_hat_dict = dict()
    for lr_ind, lr in enumerate(lr_list):
        num_epochs, run_time, x_hat = algo_graph_iht(
            x_mat=x_tr_mat, y_tr=y_tr, max_epochs=max_epochs,
            lr=lr, x0=x0, tol_algo=tol_algo, edges=edges,
            costs=costs, g=g, s=s, root=root, h_factor=h_factor,
            gamma=gamma, proj_max_num_iter=proj_max_num_iter,
            verbose=verbose)
        y_err = np.linalg.norm(y_va - np.dot(x_va_mat, x_hat)) ** 2.
        test_err_mat[lr_ind] = y_err
        x_hat_dict[lr] = (num_epochs, run_time, x_hat)
    min_index = np.argmin(test_err_mat)
    best_lr = lr_list[min_index]
    err = np.linalg.norm(x_star - x_hat_dict[best_lr][2])
    num_epochs, run_time = x_hat_dict[best_lr][:2]
    return err, num_epochs, run_time


def algo_graph_sto_iht(
        x_mat, y_tr, max_epochs, lr, x0, tol_algo, edges, costs, g,
        root, s, h_factor, gamma, proj_max_num_iter, verbose, b):
    np.random.seed()
    start_time = time.time()
    x_hat = x0
    (n, p) = x_mat.shape
    x_tr_t = np.transpose(x_mat)
    b = n if n < b else b
    num_blocks = int(n) / int(b)
    prob = [1. / num_blocks] * num_blocks

    # graph info
    h_low = int(h_factor * s)
    h_high = int(h_factor * s * (1. + gamma))
    t_low, t_high = int(s), int(s * (1. + gamma))

    num_epochs = 0
    for epoch_i in range(max_epochs):
        num_epochs += 1
        for _ in range(num_blocks):
            ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            xtx = np.dot(x_tr_t[:, block], x_mat[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            gradient = -2. * (xty - np.dot(xtx, x_hat))
            head_nodes, proj_grad = algo_head_tail_binsearch(
                edges, gradient, costs, g, root, h_low, h_high,
                proj_max_num_iter, verbose)
            bt = x_hat - (lr / (prob[ii] * num_blocks)) * proj_grad
            tail_nodes, proj_bt = algo_head_tail_binsearch(
                edges, bt, costs, g, root,
                t_low, t_high, proj_max_num_iter, verbose)
            x_hat = proj_bt
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return num_epochs, run_time, x_hat


def cv_graph_sto_iht(x_tr_mat, y_tr, x_va_mat, y_va, b_list, lr_list, s,
                     max_epochs, tol_algo, x_star, x0, verbose, edges, costs,
                     g, root, proj_max_num_iter, h_factor, gamma):
    """ Tuning parameter by using additional validation dataset. """
    test_err_mat = np.zeros(len(lr_list) * len(b_list))
    para_dict = dict()
    x_hat_dict = dict()
    for index, (lr, b) in enumerate(product(lr_list, b_list)):
        num_epochs, run_time, x_hat = algo_graph_sto_iht(
            x_mat=x_tr_mat, y_tr=y_tr, max_epochs=max_epochs,
            lr=lr, x0=x0, tol_algo=tol_algo, edges=edges,
            costs=costs, g=g, root=root, s=s, h_factor=h_factor,
            gamma=gamma, proj_max_num_iter=proj_max_num_iter,
            verbose=verbose, b=b)
        y_err = np.linalg.norm(y_va - np.dot(x_va_mat, x_hat)) ** 2.
        test_err_mat[index] = y_err
        para_dict[index] = (lr, b)
        x_hat_dict[(lr, b)] = (num_epochs, run_time, x_hat)
    min_index = np.argmin(test_err_mat)
    lr, b = para_dict[min_index]
    err = np.linalg.norm(x_star - x_hat_dict[(lr, b)][2])
    num_epochs, run_time = x_hat_dict[(lr, b)][:2]
    return err, num_epochs, run_time


def algo_niht(x_mat, y_tr, max_epochs, s, x_star, x0, tol_algo):
    start_time = time.time()
    x_hat = x0
    c = 0.01
    kappa = 2. / (1 - c)
    (m, p) = x_mat.shape
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y_tr)
    gamma = np.argsort(np.abs(np.dot(x_tr_t, y_tr)))[-s:]

    num_epochs = 0
    for epoch_i in range(max_epochs):
        num_epochs += 1
        # we obey the implementation used in their code
        gn = xty - np.dot(xtx, x_hat)
        tmp_v = np.dot(x_mat[:, gamma], gn[gamma])
        xx = np.dot(gn[gamma], gn[gamma])
        yy = np.dot(tmp_v, tmp_v)
        if yy != 0:
            mu = xx / yy
        else:
            mu = 1.
        bt = x_hat + mu * gn
        bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
        w_tmp = bt
        gamma_next = np.nonzero(w_tmp)[0]
        if set(gamma_next).__eq__(set(gamma)):
            x_hat = w_tmp
        else:
            xx = np.linalg.norm(w_tmp - x_hat) ** 2.
            yy = np.linalg.norm(np.dot(x_mat, w_tmp - x_hat)) ** 2.
            if yy <= 0.0:
                continue
            if mu <= (1. - c) * xx / yy:
                x_hat = w_tmp
            elif mu > (1. - c) * xx / yy:
                while True:
                    mu = mu / (kappa * (1. - c))
                    bt = x_hat + mu * gn
                    bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
                    w_tmp = bt
                    xx = np.linalg.norm(w_tmp - x_hat) ** 2.
                    yy = np.linalg.norm(np.dot(x_mat, w_tmp - x_hat)) ** 2.
                    if yy <= 0.0:
                        break
                    if mu <= (1 - c) * xx / yy:
                        break
                gamma_next = np.nonzero(w_tmp)[0]
                x_hat = w_tmp
                gamma = gamma_next

        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
    x_err = np.linalg.norm(x_hat - x_star)
    run_time = time.time() - start_time
    return x_err, num_epochs, run_time


def algo_graph_cosamp(
        x_mat, y_tr, max_epochs, x_star, x0, tol_algo, edges, costs,
        h_g, t_g, root, h_low, h_high, t_low, t_high, proj_max_num_iter,
        verbose):
    start_time = time.time()
    x_hat = np.zeros_like(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y_tr)
    num_epochs = 0
    y_err_list = []
    x_err_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        grad = -2. * (np.dot(xtx, x_hat) - xty)  # proxy
        head_nodes, proj_grad = algo_head_tail_binsearch(
            edges, grad, costs, h_g, root,
            h_low, h_high, proj_max_num_iter, verbose)
        gamma = np.union1d(x_hat.nonzero()[0], head_nodes)
        bt = np.zeros_like(x_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_mat[:, gamma]), y_tr)
        tail_nodes, proj_bt = algo_head_tail_binsearch(
            edges, bt, costs, t_g, root,
            t_low, t_high, proj_max_num_iter, verbose)
        x_hat = proj_bt

        y_err_list.append(np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) ** 2.)
        x_err_list.append(np.linalg.norm(x_hat - x_star))

        if np.linalg.norm(x_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break

        if verbose > 0:
            print(epoch_i, t_low, x_err_list[-1])
    x_err = np.linalg.norm(x_hat - x_star)
    run_time = time.time() - start_time
    return x_err, num_epochs, run_time


def algo_cosamp(x_mat, y_tr, max_epochs, x_star, x0, tol_algo, s):
    start_time = time.time()
    x_hat = np.zeros_like(x0)
    x_tr_t = np.transpose(x_mat)
    m, p = x_mat.shape
    num_epochs = 0
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y_tr)
    for epoch_i in range(max_epochs):
        num_epochs += 1
        grad = -(2. / float(m)) * (np.dot(xtx, x_hat) - xty)  # proxy
        gamma = np.argsort(abs(grad))[-2 * s:]  # identify
        gamma = np.union1d(x_hat.nonzero()[0], gamma)
        bt = np.zeros_like(x_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_mat[:, gamma]), y_tr)
        gamma = np.argsort(abs(bt))[-s:]
        x_hat = np.zeros_like(x_hat)
        x_hat[gamma] = bt[gamma]

        if np.linalg.norm(x_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
    x_err = np.linalg.norm(x_hat - x_star)
    run_time = time.time() - start_time
    return x_err, num_epochs, run_time


grid_data = {
    'subgraph': [93, 94, 95, 96, 97,
                 123, 124, 125, 126, 127,
                 153, 154, 155, 156, 157,
                 183, 184, 185, 186, 187,
                 213, 214, 215, 216, 217],
    # grid size (length).
    'height': 30,
    # grid width (width).
    'width': 30,
    # the dimension of grid graph is 33 x 33.
    'p': 30 * 30,
    # sparsity list
    's': 25
}


def print_helper(method, trial_i, n, err, num_epochs, run_time):
    print('%-13s trial_%03d n: %03d w_error: %.3e '
          'num_epochs: %03d run_time: %.3e' %
          (method, trial_i, n, err, num_epochs, run_time))


def run_single_test(data):
    np.random.seed()
    s, n, p, x_star = data['s'], data['n'], data['p'], data['x_star']
    x0, tol_algo, tol_rec = data['x0'], data['tol_algo'], data['tol_rec']
    max_epochs = data['max_epochs']
    b_list, lr_list = data['b_list'], data['lr_list']
    verbose = data['verbose']
    trial_i = data['trial_i']
    x_tr_mat = np.random.normal(0.0, 1.0, size=n * p)
    x_va_mat = np.random.normal(0.0, 1.0, size=100 * p)
    x_tr_mat = np.reshape(x_tr_mat, (n, p)) / np.sqrt(n)
    x_va_mat = np.reshape(x_va_mat, (100, p)) / np.sqrt(100.0)
    y_tr = np.dot(x_tr_mat, x_star)
    y_va = np.dot(x_va_mat, x_star)

    # graph information and projection parameters
    gamma = data['proj_para']['gamma']
    edges, costs = data['proj_para']['edges'], data['proj_para']['costs']
    g, root = data['proj_para']['g'], data['proj_para']['root']
    h_factor = data['proj_para']['h_factor'][s]
    proj_max_num_iter = data['proj_para']['proj_max_num_iter']
    rec_err = []

    # ------------ NIHT -----------
    err, num_epochs, run_time = algo_niht(
        x_mat=x_tr_mat, y_tr=y_tr, max_epochs=max_epochs, s=s, x_star=x_star,
        x0=x0, tol_algo=tol_algo)
    rec_err.append(('niht', err))
    print_helper('niht', trial_i, n, err, num_epochs, run_time)

    # ------------ IHT ------------
    err, num_epochs, run_time = cv_iht(
        x_tr_mat=x_tr_mat, y_tr=y_tr, x_va_mat=x_va_mat, y_va=y_va,
        max_epochs=max_epochs, lr_list=lr_list,
        s=s, x_star=x_star, x0=x0, tol_algo=tol_algo)
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
        max_epochs=max_epochs, lr_list=lr_list,
        x_star=x_star, x0=x0, tol_algo=tol_algo, edges=edges,
        costs=costs, g=g, root=root, s=s, h_factor=h_factor,
        gamma=gamma, proj_max_num_iter=proj_max_num_iter, verbose=verbose)
    rec_err.append(('graph-iht', err))
    print_helper('graph-iht', trial_i, n, err, num_epochs, run_time)

    # ------------ GraphStoIHT ------------
    err, num_epochs, run_time = cv_graph_sto_iht(
        x_tr_mat, y_tr, x_va_mat, y_va, b_list=b_list, lr_list=lr_list, s=s,
        max_epochs=max_epochs, tol_algo=tol_algo, x0=x0, x_star=x_star,
        edges=edges, costs=costs, g=g, root=root, h_factor=h_factor,
        gamma=gamma, proj_max_num_iter=proj_max_num_iter, verbose=verbose)
    rec_err.append(('graph-sto-iht', err))
    print_helper('graph-sto-iht', trial_i, n, err, num_epochs, run_time)

    # ------------ GraphCoSaMP ------------
    h_low, h_high = int(2 * s), int(2 * s * (1.0 + gamma))
    t_low, t_high = int(s), int(s * (1.0 + gamma))
    err, num_epochs, run_time = algo_graph_cosamp(
        x_mat=x_tr_mat, y_tr=y_tr, max_epochs=max_epochs, x_star=x_star,
        x0=x0, tol_algo=tol_algo, edges=edges, costs=costs,
        h_g=1, t_g=1, root=root, h_low=h_low, h_high=h_high,
        t_low=t_low, t_high=t_high, proj_max_num_iter=proj_max_num_iter,
        verbose=verbose)
    rec_err.append(('graph-cosamp', err))
    print_helper('graph-cosamp', trial_i, n, err, num_epochs, run_time)

    # ------------ CoSaMP ------------
    err, num_epochs, run_time = algo_cosamp(
        x_mat=x_tr_mat, y_tr=y_tr, max_epochs=max_epochs, x_star=x_star,
        x0=x0, tol_algo=tol_algo, s=s)
    rec_err.append(('cosamp', err))
    print_helper('cosamp', trial_i, n, err, num_epochs, run_time)
    return trial_i, n, rec_err


def run_test(trial_range, n_list, tol_algo, tol_rec, max_epochs, num_cpus):
    np.random.seed()
    start_time = time.time()
    input_data_list, saved_data = [], dict()
    for trial_i in trial_range:
        for n in n_list:
            height = grid_data['height']
            width = grid_data['width']
            edges, costs = simu_grid_graph(height=height, width=width)
            p, s = grid_data['p'], grid_data['s']
            x_star = np.zeros(p)  # using Gaussian signal.
            x_star[grid_data['subgraph']] = np.random.normal(0.0, 1.0, size=s)
            data = {
                'trial_i': trial_i,
                's': s,
                'n': n,
                'p': p,
                'max_epochs': max_epochs,
                'n_list': n_list,
                'lr_list': [0.2, 0.4, 0.6, 0.8],
                'b_list': [int(n) / 5, int(n) / 10],
                'num_fold': 10,
                'x_star': x_star,
                'x0': np.zeros(p),
                'subgraph': grid_data['subgraph'],
                'tol_algo': tol_algo,
                'height': grid_data['height'],
                'width': grid_data['width'],
                'tol_rec': tol_rec,
                'grid_data': grid_data,
                'verbose': 0,
                'proj_para': {
                    'edges': edges,
                    'costs': costs,
                    'g': 1,
                    'root': -1,
                    'proj_max_num_iter': 50,
                    'verbose': 0,
                    'gamma': 0.1,
                    'h_factor': {
                        s: h_factor for s, h_factor in
                        zip([grid_data['s']], [20.0])
                    }
                }
            }
            if trial_i not in saved_data:
                saved_data[trial_i] = data
            input_data_list.append(data)
    pool = multiprocessing.Pool(processes=num_cpus)
    results_pool = pool.map(run_single_test, input_data_list)
    pool.close()
    pool.join()

    sum_results = dict()  # trial_i, n, rec_err
    for trial_i, n, rec_err in results_pool:
        if trial_i not in sum_results:
            sum_results[trial_i] = []
        sum_results[trial_i].append((trial_i, n, rec_err))
    for trial_i in sum_results:
        f_name = root_p + 'sr_simu_test05_trial_%02d.pkl' % trial_i
        print('save results to file: %s' % f_name)
        pickle.dump({'results_pool': sum_results[trial_i]},
                    open(f_name, 'wb'))
    print('total run time of %02d trials: %.2f seconds.' %
          (len(trial_range), time.time() - start_time))


def summarize_results(trials_range, n_list, method_list, tol_rec):
    results_pool = []
    num_trials = len(trials_range)
    for trial_i in trials_range:
        f_name = root_p + 'sr_simu_test05_trial_%02d.pkl' % trial_i
        print('load file: %s' % f_name)
        for item in pickle.load(open(f_name))['results_pool']:
            results_pool.append(item)
    sum_results = {_: np.zeros((num_trials, len(n_list))) for _ in method_list}
    for trial_i, n, rec_err in results_pool:
        n_ind = list(n_list).index(n)
        for method, err in rec_err:
            ind = list(trials_range).index(trial_i)
            sum_results[method][ind][n_ind] = err
    num_trim = int(round(0.05 * num_trials))
    trim_results = dict()
    for method in method_list:
        re = sum_results[method]
        # remove 5% best and 5% worst.
        trim_re = np.sort(re, axis=0)[num_trim:num_trials - num_trim, :]
        re = trim_re
        re[re > tol_rec] = 0.
        re[re != 0.0] = 1.0
        trim_re = re
        trim_re = np.mean(trim_re, axis=0)
        trim_results[method] = trim_re
    f_name = root_p + 'sr_grid.pkl'
    print('save file to: %s' % f_name)
    pickle.dump({'trim_results': trim_results,
                 'sum_result': sum_results,
                 'results_pool': results_pool}, open(f_name, 'wb'))


def show_test(n_list, method_list, method_label_list):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 8, 3.5
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18

    color_list = ['c', 'b', 'g', 'k', 'm', 'y', 'r']
    marker_list = ['D', 'X', 'o', 'h', 'P', 'p', 's']
    linestyle_list = ['-', '-', '-', '-', '-', '-', '-']

    fig, ax = plt.subplots(1, 2)
    ax[1].grid(b=True, which='both', color='lightgray',
               linestyle='dotted', axis='both')
    ax[1].set_xticks([1, 2, 3, 4, 5, 6])
    ax[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1].set_xlim([1., 6.])
    ax[1].set_ylim([-0.03, 1.03])
    height, width = grid_data['height'], grid_data['width']
    x_star = np.ones((height, width)) * np.nan
    for node_id in grid_data['subgraph']:
        i, j = node_id / height, node_id % width
        x_star[i, j] = np.random.normal(0.0, 1.0)
    for i in range(height + 1):
        ax[0].axhline(i, lw=.05, color='gray', linestyle='dotted',
                      zorder=5)
    for i in range(width + 1):
        ax[0].axvline(i, lw=.05, color='gray', linestyle='dotted',
                      zorder=5)
    ax[0].imshow(x_star, interpolation='none', cmap='gray',
                 extent=[0, height, 0, width], zorder=0)
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].axes.get_yaxis().set_visible(False)
    results = pickle.load(open(root_p + 'sr_grid.pkl'))
    trim_results = results['trim_results']
    for method_ind, method in enumerate(method_list):
        ax[1].plot(np.asarray(n_list) / float(grid_data['s']),
                   trim_results[method],
                   c=color_list[method_ind],
                   linestyle=linestyle_list[method_ind],
                   markerfacecolor='none',
                   marker=marker_list[method_ind], markersize=5.,
                   markeredgewidth=1.0, linewidth=1.0,
                   label=r"$\displaystyle \textsc{%s}$ " %
                         method_label_list[method_ind])
    ax[1].set_xlabel('Oversampling ratio ' + r"$\displaystyle n/s$ ",
                     labelpad=0)
    ax[1].set_ylabel('Probability of Recovery', labelpad=0.5)
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax[1].legend(loc='lower right', bbox_to_anchor=(1.3, 0.01),
                 fontsize=11., frameon=True, borderpad=0.1,
                 labelspacing=0.2, handletextpad=0.1, markerfirst=True)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    plt.subplots_adjust(wspace=0.25, hspace=0.0)
    f_name = root_p + 'sr_grid.pdf'
    print('save fig to: %s' % f_name)
    fig.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0,
                format='pdf')
    plt.close()


def main():
    num_trials = 50
    max_epochs = 500
    tol_algo = 1e-7
    tol_rec = 1e-6
    command = sys.argv[1]
    n_list = range(20, 151, 5)
    method_list = ['niht', 'iht', 'sto-iht', 'cosamp',
                   'graph-iht', 'graph-cosamp', 'graph-sto-iht']
    if command == 'run_test':
        num_cpus = int(sys.argv[2])
        trial_range = range(int(sys.argv[3]), int(sys.argv[4]))
        run_test(trial_range=trial_range,
                 n_list=n_list,
                 tol_algo=tol_algo,
                 tol_rec=tol_rec,
                 max_epochs=max_epochs,
                 num_cpus=num_cpus)
    elif command == 'summarize_results':
        trials_range = range(50)
        summarize_results(trials_range=trials_range, n_list=n_list,
                          method_list=method_list, tol_rec=tol_rec)
    elif command == 'show_test':
        method_label_list = ['NIHT', 'IHT', 'StoIHT', 'CoSaMP',
                             'GraphIHT', 'GraphCoSaMP', 'GraphStoIHT']
        show_test(n_list=n_list, method_list=method_list,
                  method_label_list=method_label_list)


if __name__ == '__main__':
    main()