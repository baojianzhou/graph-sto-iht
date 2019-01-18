# -*- coding: utf-8 -*-
import os
import time
import pickle
from os import sys, path
import numpy as np
from numpy.linalg import norm
import multiprocessing
from itertools import product

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


def simu_grid_graph(width, height):
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
    weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights


def thresh(x, k):
    """
    K-sparse projection algorithm.
    :param x: the vector to be projected.
    :param k: sparse parameter k
    :return: return a binary vector, supp_x
            where the supp_x_i = 1. if |x_i| is in top k
            among |x_i|, i = 0,1,.., n-1. supp_x_i = 0 otherwise.
    """
    start_time = time.time()
    indices = np.argsort(np.abs(x))[::-1]  # by using descending order
    supp_x = np.zeros_like(x)
    supp_x[indices[:k]] = 1.
    time_proj = time.time() - start_time
    return supp_x, time_proj


def generate_dataset(root):
    import scipy.io as sio
    import networkx as nx
    p = 100 * 100
    edges, weis = simu_grid_graph(width=100, height=100)
    g = nx.Graph()
    [g.add_edge(edge[0], edge[1]) for edge in edges]
    # 415 nodes, 4 components
    img_icml = sio.loadmat(root + 'image_icml.mat')['x_gray']
    w_icml = img_icml.flatten()
    sub_graph = np.nonzero(w_icml)[0]
    num_cc = nx.number_connected_components(g.subgraph(sub_graph))
    print('icml has %d nodes and %d components' % (len(sub_graph), num_cc))
    # 622 nodes, 1 component
    img_angio = sio.loadmat(root + 'image_angio.mat')['x_gray']
    w_angio = img_angio.flatten()
    sub_graph = np.nonzero(w_angio)[0]
    num_cc = nx.number_connected_components(g.subgraph(sub_graph))
    print('angio has %d nodes and %d components' % (len(sub_graph), num_cc))
    # 613 nodes, 3 components
    img_background = sio.loadmat(root + 'image_background.mat')['x_gray']
    w_background = img_background.flatten()
    sub_graph = np.nonzero(w_background)[0]
    num_cc = nx.number_connected_components(g.subgraph(sub_graph))
    print('background has %d nodes and %d components' % (
        len(sub_graph), num_cc))
    return p, w_icml, w_angio, w_background


def get_model_space(x_model, c):
    models = {
        'cosamp': {
            'select_atom': 2,
            'gamma': 0.1,
            'tol': 1e-4,
            'max_iter': 50,
            'verbose': 0,
            'tol_early': 5e-3,
            'debias': 0,
            'hhs': 0,
            'im_height': 100,
            'im_width': 100,
            'display_perf': 1,
            'n': len(x_model),
            'k': len(np.nonzero(x_model)[0]),
            'c': c,
        },
        'model_cs': {
            'select_atom': 2,
            'gamma': 0.1,
            'tol': 1e-4,
            'max_iter': 50,
            'verbose': 0,
            'tol_early': 5e-3,
            'debias': 0,
            'hhs': 0,
            'im_height': 100,
            'im_width': 100,
            'display_perf': 1,
            'n': len(x_model),
            'k': len(np.nonzero(x_model)[0]),
            'c': c, },
    }


def mat_sub_f(mat_f_, x, indices, n):
    """ assume consistency all around this function takes as input
        a) function handle A \in R^{MxN}
        b) vector x \in R^{length(I)x1} and computes y = A(:,I)*x
    :param mat_f_: matrix lambda function. like handle in Matlab.
    :param x:
    :param indices:
    :param n:
    :return:
    """
    z = np.zeros(n)
    z[indices] = x
    y = mat_f_(z)
    return y


def mat_sub_trans_f(mat_trans_f_, x, indices):
    """ assume consistency all around this function takes as a vector x
        and computes y = At_f(:,indices)*x
    :param mat_trans_f_:
    :param x:
    :param indices:
    :return:
    """
    z = mat_trans_f_(x)
    y = z[indices]
    return y


def mat_f(x, omega_, perm_=None):
    """
    :param x: n elements vector of an input.
    :param omega_: k/2 vector denoting which Fourier coefficients to use
            ( the real and imag parts of each freq are kept.)
    :param perm_: Permutation to apply to the input vector.  Fourier coeffs of
            x(P) are calculated. Default = 1:N (no scrambling).
    :return:
    """
    p = len(x)
    if perm_ is None:
        perm_ = range(p)
    fx = (1. / np.sqrt(p)) * np.fft.fft(x[perm_])
    real_part = np.sqrt(2) * np.real(fx[omega_])
    img_part = np.sqrt(2) * np.imag(fx[omega_])
    b = np.concatenate([real_part, img_part])
    return b


def mat_trans_f(b, n_, omega_, perm_=None):
    """
    :param b: k vector = [real part; imag part]
    :param n_: length of output x
    :param omega_: k/2 vector denoting which Fourier coefficients to use
            ( the real and imag parts of each freq are kept.)
    :param perm_: Permutation to apply to the input vector.  Fourier coeffs of
            x(P) are calculated. Default = 1:N (no scrambling).
    :return:
    """
    if perm_ is None:
        perm_ = range(n_)
    k = len(b)
    fx = np.zeros(n_, dtype=complex)
    fx[omega_] = np.sqrt(2) * b[:k / 2] + np.sqrt(2) * b[k / 2:] * 1j
    x = np.zeros(n_)
    x[perm_] = np.sqrt(n_) * np.real(np.fft.ifft(fx))
    return x


def mat_block_f(x, block, omega, perm_=None):
    """
    :param x: n elements vector of an input.
    :param block:
    :param omega:
    :param perm_: Permutation to apply to the input vector.  Fourier coeffs of
            x(P) are calculated. Default = 1:N (no scrambling).
    :return:
    """
    n = len(x)
    sub_indices = omega[block]
    if perm_ is None:
        perm_ = range(n)
    fx = (1. / np.sqrt(n)) * np.fft.fft(x[perm_])
    real_part = np.sqrt(2) * np.real(fx[sub_indices])
    imag_part = np.sqrt(2) * np.imag(fx[sub_indices])
    b = np.concatenate([real_part, imag_part])
    return b


def mat_block_trans_f(b, n_, block, omega, perm_=None):
    """
    :param b: k vector = [real part; imag part]
    :param n_: length of output x
    :param omega:
    :param block: k/2 vector denoting which Fourier coefficients to use
            ( the real and imag parts of each freq are kept.)
    :param perm_: Permutation to apply to the input vector.  Fourier coeffs of
            x(P) are calculated. Default = 1:N (no scrambling).
    :return:
    """
    if perm_ is None:
        perm_ = range(n_)
    k = len(b)
    sub_indices = omega[block]
    fx = np.zeros(n_, dtype=complex)
    fx[sub_indices] = np.sqrt(2) * b[:k / 2] + np.sqrt(2) * b[k / 2:] * 1j
    x = np.zeros(n_)
    x[perm_] = np.sqrt(n_) * np.real(np.fft.ifft(fx))
    return x


def cluster_k_approx(coeffs, k, c, gamma, edges, costs, verbose, use_cpp=True):
    """
    :param coeffs: squared of coefficients of signal in an array
    :param k: target sparsity
    :param c: target number of clusters
    :param gamma:
    :param edges:
    :param costs:
    :param verbose: display progress
    :param use_cpp
    :return:
    """
    t_proj_start_time = time.time()
    if use_cpp:
        opts = {'pruning': 'gw', 'max_num_iter': 50, 'verbose': 0}
        if verbose:
            opts['verbose'] = 1
        s_low = int(k)
        s_high = int(np.round((1 + gamma) * k))
        max_num_iter = opts['max_num_iter']
        verbose = opts['verbose']
        proj_nodes = wrap_head_tail_binsearch(
            edges, coeffs, costs, c, -1, s_low, s_high, max_num_iter, verbose)
        x = np.zeros_like(coeffs)
        x[proj_nodes] = 1.
        t_proj = time.time() - t_proj_start_time
    else:
        x = 0.0
        t_proj = time.time() - t_proj_start_time
    return x, t_proj


def conjugate_gradient_solve(mat_f_, b, tol, max_iter, verbose=None):
    if verbose is None:
        verbose = 1
    x = np.zeros(len(b))
    r = np.copy(b)
    d = np.copy(r)
    delta = np.dot(r, r)
    delta0 = np.dot(b, b)
    num_iter = 0
    best_x = np.copy(x)
    best_res = np.sqrt(delta / delta0)
    while (num_iter < max_iter) & (delta > (tol ** 2. * delta0)):
        q = mat_f_(d)
        alpha = delta / (np.dot(d, q))
        x += alpha * d
        if np.mod(num_iter + 1, 50) == 0:
            r = b - mat_f_(x)
        else:
            r = r - alpha * q
        delta_old = delta
        delta = np.dot(r, r)
        beta = delta / delta_old
        d = r + beta * d
        num_iter += 1
        if np.sqrt(delta / delta0) < best_res:
            best_x = x
            best_res = np.sqrt(delta / delta0)
    if verbose and (np.mod(num_iter, verbose) == 0):
        print('cg: Iter = %d, Best residual = %8.3e, Current residual = %8.3e'
              % (num_iter, best_res, np.sqrt(delta / delta0)))
    return best_x, best_res, num_iter


def cosamp_cluster(
        rec_flag, b, k, c, mat_f_, mat_trans_f_, opts, x_gray, edges, costs):
    start_time = time.time()
    n = len(mat_trans_f_(b))
    err = 1.0
    x_hat = np.zeros(n)  # to save the current solution.
    s_cosamp = np.zeros(n)  # current sparse solution.
    alpha = opts['select_atom']  # to control the sparsity at proxy stage
    verbose = opts['verbose']
    tol_early = opts['tol_early']
    tol = opts['tol']
    max_iter = opts['max_iter']
    supp_x = np.zeros_like(x_hat)
    iter_cnt = 0  # number of iteration.
    time_model = 0  # run time of sparse model projection.
    time_proj = 0  # run time of projection
    time_cgs = 0  # run time of conjugate gradient.
    while err > tol and iter_cnt <= max_iter:
        iter_cnt += 1
        residue = b - mat_f_(s_cosamp)
        proxy = mat_trans_f_(residue)
        # ------------------ estimation ------------------
        k_prime = int(round(alpha * k))
        c_prime = int(np.round(alpha * c))
        if rec_flag == 'sparse':
            t_start_model = time.time()
            supp_x, t_proj_ = thresh(proxy, k_prime)
            time_model += time.time() - t_start_model
            time_proj += t_proj_
        if rec_flag == 'cluster':
            t_start_model = time.time()
            gamma = opts['gamma']
            coeffs = proxy * proxy
            supp_x, t_proj_ = cluster_k_approx(
                coeffs, k_prime, c_prime, gamma, edges, costs, verbose)
            time_model += time.time() - t_start_model
            time_proj += t_proj_
        else:  # other sparsity algorithms.
            t_start_model = time.time()
            supp_x, t_proj_ = thresh(proxy, int(round(alpha * k)))
            time_model += time.time() - t_start_model
            time_proj += t_proj_
        # ------------------------------------------------
        if opts['hhs']:
            proj_indices = np.nonzero(supp_x)[0]
            samples = residue
        else:
            pre_indices = np.nonzero(s_cosamp)[0]
            proj_indices = np.union1d(pre_indices, np.nonzero(supp_x)[0])
            samples = b
        # -------------- least square step----------------
        pp_tt = lambda x: mat_sub_f(mat_f_, x, proj_indices, n)
        pp_trans_tt = lambda x: mat_sub_trans_f(mat_trans_f_, x, proj_indices)
        qq = pp_trans_tt(samples)
        pp_trans_pp_tt = lambda x: pp_trans_tt(pp_tt(x))
        time_start_cg = time.time()
        w, _, _ = conjugate_gradient_solve(pp_trans_pp_tt, qq, 1e-4, 20, 0)
        time_cgs += time.time() - time_start_cg
        bb = np.zeros_like(s_cosamp)
        bb[proj_indices] = w
        if opts['hhs']:
            bb = s_cosamp + bb
        # ------------------- prune stage ----------------
        if rec_flag == 'sparse':
            t_start_model = time.time()
            supp_x, t_proj_ = thresh(bb, k)
            time_model += time.time() - t_start_model
            time_proj += t_proj_
        elif rec_flag == 'cluster':
            time_start_model = time.time()
            gamma = opts['gamma']
            alpha = opts['select_atom']
            coeffs = bb * bb
            supp_x, t_proj_ = cluster_k_approx(
                coeffs, k, c, gamma, edges, costs, verbose)
            time_model += (time.time() - time_start_model)
            time_proj += time_proj + t_proj_
        else:
            t_start_model = time.time()
            supp_x, t_proj_ = thresh(bb, k)
            time_model += time.time() - t_start_model
            time_proj += t_proj_

        s_cosamp = bb * supp_x
        x_hat = s_cosamp
        # ------------------------------------------------
        x_err = norm(x_hat - x_gray) ** 2.0
        x_err /= norm(x_gray) ** 2.
        y_err = norm(mat_f_(s_cosamp) - b) / norm(b)
        if verbose:
            print('iter: %d x_err: %.4f y_err: %.4f \n' %
                  (iter_cnt, x_err, y_err))
        if x_err < 0.05:  # already there.
            break
        if (np.abs(err - y_err) / err) <= tol_early or y_err <= tol_early:
            break
        err = y_err

    if opts['debias']:
        proj_indices = np.nonzero(supp_x)[0]
        pp_tt = lambda x: mat_sub_f(mat_f_, x, proj_indices, n)
        pp_trans_tt = lambda x: mat_sub_trans_f(mat_trans_f_, x, proj_indices)
        qq = pp_trans_tt(b)
        pp_trans_pp_tt = lambda x: pp_trans_tt(pp_tt(x))
        time_start_cg = time.time()
        w, _, _ = conjugate_gradient_solve(pp_trans_pp_tt, qq, 1e-4, 20, 0)
        time_cgs += time.time() - time_start_cg
        x_hat = np.zeros_like(s_cosamp)
        x_hat[proj_indices] = w
    else:
        x_hat = x_hat
    run_time = time.time() - start_time
    return x_hat, time_cgs, time_model, time_proj, run_time


def algo_cosamp(b, k, c, mat_f_, mat_trans_f_, opts, x_gray):
    return cosamp_cluster(
        'sparse', b, k, c, mat_f_, mat_trans_f_, opts, x_gray, None, None)


def algo_graph_cosamp(
        b, k, c, mat_f_, mat_trans_f_, opts, x_gray, edges, costs):
    return cosamp_cluster(
        'cluster', b, k, c, mat_f_, mat_trans_f_, opts, x_gray, edges, costs)


def algo_iht(b, k, mat_f_, mat_trans_f_, opts, x_gray):
    w0 = np.zeros_like(x_gray)
    start_time = time.time()
    x_hat, num_iter = w0, 0
    num_epochs = 0
    lr = .8
    p = len(w0)
    s = k
    time_model, time_proj = 0.0, 0.0
    for epoch_i in range(50 * opts['max_iter']):
        num_epochs += 1
        residue = b - mat_f_(x_hat)
        grad = -1. * mat_trans_f_(residue)
        bt = x_hat - lr * grad
        bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
        x_hat = bt
        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(x_hat) >= 1e3:
            break
        x_err = norm(x_hat - x_gray) ** 2.0
        x_err /= norm(x_gray) ** 2.
        y_err = norm(mat_f_(x_hat) - b) / norm(b)
        print(x_err, y_err, len(np.nonzero(x_gray)[0]))
        if x_err < 0.05:  # already there.
            break
    run_time = time.time() - start_time
    return x_hat, 0.0, time_model, time_proj, run_time


def algo_graph_iht(y0, lr, k, c, opts, x0, edges, costs, omega):
    start_time = time.time()
    x_hat, num_iter = np.copy(x0), 0
    num_epochs = 0
    verbose = opts['verbose']
    time_model, time_proj = 0.0, 0.0
    gamma = opts['gamma']
    y_err_list = []
    for epoch_i in range(500):
        m_f = lambda x: mat_f(x, omega)
        m_trans_f = lambda x: mat_trans_f(x, len(x_hat), omega)
        num_epochs += 1
        residue = y0 - m_f(x_hat)
        grad = -2. * m_trans_f(residue)
        t_start_model = time.time()
        supp_h, t_proj_ = cluster_k_approx(
            grad * grad, len(x_hat) / 2, c, gamma, edges, costs, verbose)
        time_model += (time.time() - t_start_model)
        time_proj += t_proj_
        bt = x_hat - lr * (supp_h * grad)
        t_start_model = time.time()
        supp_t, t_proj_ = cluster_k_approx(
            bt * bt, k, c, gamma, edges, costs, verbose)
        time_model += (time.time() - t_start_model)
        time_proj += t_proj_
        x_hat = bt * supp_t

        # diverge cases because of the large learning rate: early stopping
        y_err = norm(m_f(x_hat) - y0) ** 2.
        y_err_list.append(y_err)
        if np.linalg.norm(x_hat) >= 1e3 or y_err > 1e3:
            break
        if epoch_i >= 10 and len(np.unique(y_err_list[-5:])) == 1:
            break
        if y_err <= 1e-6:
            break
    run_time = time.time() - start_time
    return x_hat, time_model, time_proj, run_time


def algo_graph_sto_iht(y0, lr, b, k, c, opts, x0, edges, costs, omega):
    start_time = time.time()
    x_hat, num_iter = np.copy(x0), 0
    num_epochs = 0
    verbose = opts['verbose']
    time_model, time_proj = 0.0, 0.0
    gamma = opts['gamma']
    num_blocks = len(omega) / b
    y_err_list = []
    for epoch_i, iter_i in product(range(500), range(num_blocks)):
        ii = np.random.randint(0, num_blocks)
        real_indices = np.asarray(range(ii * b, (ii + 1) * b))
        img_indices = np.asarray(range(ii * b, (ii + 1) * b)) + len(omega)
        omega_ = omega[real_indices]
        m_f = lambda x: mat_f(x, omega_)
        m_trans_f = lambda x: mat_trans_f(x, len(x_hat), omega_)
        num_epochs += 1
        y1 = np.zeros(2 * len(omega_))
        y1[:len(omega_)] = y0[real_indices]  # real part
        y1[len(omega_):] = y0[img_indices]
        residue = y1 - m_f(x_hat)
        grad = -2. * m_trans_f(residue)
        t_start_model = time.time()
        supp_h, t_proj_ = cluster_k_approx(
            grad * grad, len(x_hat) / 2, c, gamma, edges, costs, verbose)
        time_model += (time.time() - t_start_model)
        time_proj += t_proj_
        bt = x_hat - lr * (supp_h * grad)
        t_start_model = time.time()
        supp_t, t_proj_ = cluster_k_approx(
            bt * bt, k, c, gamma, edges, costs, verbose)
        time_model += (time.time() - t_start_model)
        time_proj += t_proj_
        x_hat = bt * supp_t

        # diverge cases because of the large learning rate: early stopping
        y_err = norm(mat_f(x_hat, omega) - y0) ** 2.
        y_err_list.append(y_err)
        if np.linalg.norm(x_hat) >= 1e3 or y_err > 1e3:
            break
        if epoch_i >= 10 and len(np.unique(y_err_list[-5:])) == 1:
            break
        if y_err <= 1e-6:
            break
    run_time = time.time() - start_time
    return x_hat, time_model, time_proj, run_time


def cv_graph_sto_iht(y0, y_va, omega_va,
                     k, c, opts, x_gray, edges, costs, omega):
    """ Tuning parameter by using additional validation dataset. """
    start_time = time.time()
    lr_list = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
    b_list = [len(omega) / 5, len(omega), len(omega) / 10]
    test_err_mat = np.ones(len(lr_list) * len(b_list))
    para_dict = dict()
    x_hat_dict = dict()
    x_err_dict = dict()
    x0 = np.zeros_like(x_gray)
    for index, (lr, b) in enumerate(product(lr_list, b_list)):
        x_hat, time_model, time_proj, run_time = algo_graph_sto_iht(
            y0, lr, b, k, c, opts, x0, edges, costs, omega)
        y_err = norm(mat_f(x_hat, omega_va) - y_va) ** 2.
        test_err_mat[index] = y_err
        para_dict[index] = (lr, b)
        x_hat_dict[(lr, b)] = x_hat
        x_err = np.linalg.norm(x_hat - x_gray) ** 2.0
        x_err = x_err / (np.linalg.norm(x_gray) ** 2.0)
        x_err_dict[(lr, b)] = x_err
        if x_err < 0.05:
            break
    lr, b = para_dict[np.argmin(test_err_mat)]
    run_time = time.time()
    return x_err_dict[(lr, b)], run_time - start_time


def cv_graph_iht(y0, y_va, omega_va, k, c, opts, x_gray, edges, costs, omega):
    """ Tuning parameter by using additional validation dataset. """
    start_time = time.time()
    lr_list = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
    test_err_mat = np.ones(len(lr_list))
    para_dict = dict()
    x_hat_dict = dict()
    x_err_dict = dict()
    x0 = np.zeros_like(x_gray)
    for index, lr in enumerate(lr_list):
        x_hat, time_model, time_proj, run_time = algo_graph_iht(
            y0, lr, k, c, opts, x0, edges, costs, omega)
        y_err = norm(mat_f(x_hat, omega_va) - y_va) ** 2.
        test_err_mat[index] = y_err
        para_dict[index] = lr
        x_hat_dict[lr] = x_hat
        x_err = np.linalg.norm(x_hat - x_gray) ** 2.0
        x_err = x_err / (np.linalg.norm(x_gray) ** 2.0)
        x_err_dict[lr] = x_err
        if x_err < 0.05:
            break
    lr = para_dict[np.argmin(test_err_mat)]
    run_time = time.time()
    return x_err_dict[lr], run_time - start_time


def test_single_algo():
    root = '/network/rit/lab/ceashpc/bz383376/data/icml19/images/'
    import scipy.io as sio
    img_icml = sio.loadmat(root + 'image_icml.mat')['x_gray']
    w = img_icml.flatten()
    x_gray = w
    n = 10000
    k = len(np.nonzero(x_gray)[0])
    m = int(2 * round(2.8 * k / 2))
    perm = np.arange(0, 10000)
    q = np.random.permutation(n / 2 - 1) + 1
    omega = q[:m / 2]
    a = lambda x: mat_f(x, omega, perm)
    at = lambda x: mat_trans_f(x, n, omega, perm)
    sg_var = 0.0
    y0 = a(x_gray)
    yn = sg_var * np.random.randn(np.size(y0))
    y = y0 + yn
    rec_flag = 'cluster'
    c = 4
    opts = {'select_atom': 1,
            'gamma': 0.1000,
            'tol': 1e-6,
            'max_iter': 50,
            'verbose': 0,
            'tol_early': 1e-6,
            'debias': 0,
            'hhs': 0,
            'imheight': 100,
            'imwidth': 100,
            'display_perf': 1}

    edges, costs = simu_grid_graph(width=100, height=100)
    cosamp_cluster(rec_flag, y, k, c, a, at, opts, x_gray, edges, costs)


def edge4_index(h, w):
    n = h * w
    edge4 = np.zeros((n, 5))
    # self
    edge4[:, 0] = range(n)
    # down
    is_ = edge4[:, 0] + 1
    is_[h - 1:n:h] = range(h - 2, n - 1, h)
    edge4[:, 1] = is_
    # up
    is_ = edge4[:, 0] - 1
    is_[0:n - h + 1:h] = range(1, n - h + 2, h)
    edge4[:, 2] = is_
    # left
    is_ = edge4[:, 0] - h
    is_[:h] = range(h, 2 * h)
    edge4[:, 3] = is_
    # right
    is_ = edge4[:, 0] + h
    is_[n - h:n] = range(n - 2 * h, n - h)
    edge4[:, 4] = is_
    return edge4


def get_blocks_matrix(h, w, step):
    b = dict()
    if w == 1:
        for i in range(h):
            if i <= step:
                b[i] = range(i + step)
            elif i >= h - step:
                b[i] = range(i - step - 1, h)
            else:
                b[i] = range(i - step - 1, i + step)
        pass
    else:
        edge4 = edge4_index(h, w)
        for i in range(len(edge4)):
            b[i] = np.unique(edge4[i])
    h = len(b)
    max_num = 0
    for i in range(h):
        tmp = len(b[i])
        max_num = np.max([max_num, tmp])
    b_mat = -np.ones((h, max_num), dtype=int)
    for i in range(h):
        tmp = b[i]
        b_mat[i, :len(tmp)] = tmp
    return b, b_mat


def get_blocks_connection_matrix(bm, edge4):
    m, max_num = np.shape(bm)
    bc = dict()
    for i in range(m):
        bi = bm[i][bm[i] >= 0]
        bi_con = edge4[bi, :]
        bi_con = np.unique(bi_con.flatten())
        bi_con = bi_con[bi_con >= 0]
        tag = np.in1d(bm, bi_con)
        tag = np.reshape(tag, np.shape(bm))
        tag = np.sum(tag, axis=1)
        tag[i] = 0
        indices = np.nonzero(tag)[0]
        bci = indices
        bc[i] = bci
    max_num = 0
    for i in range(m):
        tmp = len(bc[i])
        max_num = np.max([max_num, tmp])
    bc_m = -np.ones((m, max_num), dtype=int)
    for i in range(m):
        tmp = bc[i]
        bc_m[i, :len(tmp)] = tmp
    return bc, bc_m


def run_single_algo(para):
    np.random.seed()
    trial_i, img_name, m_vec, m_ind, p, k, c, opts, x_gray = para
    result = dict()
    m = int(2 * round(m_vec[m_ind] * k / 2))
    perm = np.arange(0, p)
    q = np.random.permutation(p / 2 - 1) + 1
    omega = q[:m / 2]
    omega_va = q[m / 2:m / 2 + 100]
    m_f = lambda x: mat_f(x, omega, perm)  # Fourier operator.
    m_trans_f = lambda x: mat_trans_f(x, p, omega, perm)
    y0 = m_f(x_gray)
    y_va = mat_f(x_gray, omega_va, perm)
    y = y0
    # ---------------------- CoSaMP ----------------------------------
    re = algo_cosamp(y, k, c, m_f, m_trans_f, opts, x_gray)
    x_hat, time_cgs, time_model, time_proj, run_time = re
    err = norm(x_gray - x_hat) ** 2. / norm(x_gray) ** 2.
    print("CoSaMP      -- Trial_%02d Oversampling factor:(%.1f, m=%04d)" %
          (trial_i, m_vec[m_ind], m)),
    print('img: %-10s error: %.4f time: %.4f' % (img_name, err, run_time))
    result['cosamp'] = [err, run_time]
    # ---------------------- Graph-CoSaMP ----------------------------
    edges, costs = opts['edges'], opts['costs']
    re = algo_graph_cosamp(
        y, k, c, m_f, m_trans_f, opts, x_gray, edges, costs)
    x_hat, time_cgs, time_model, time_proj, run_time = re
    err = norm(x_gray - x_hat) ** 2. / norm(x_gray) ** 2.
    print("GraphCoSaMP -- Trial_%02d Oversampling factor:(%.1f, m=%04d)" %
          (trial_i, m_vec[m_ind], m)),
    print('img: %-10s error: %.4f time: %.4f' % (img_name, err, run_time))
    result['graph-cosamp'] = [err, run_time]
    # ----------------------- Graph-IHT --------------------------
    err, run_time = cv_graph_iht(y0, y_va, omega_va,
                                 k, c, opts, x_gray, edges, costs, omega)
    print("GraphIHT    -- Trial_%02d Oversampling factor:(%.1f, m=%04d)" %
          (trial_i, m_vec[m_ind], m)),
    print('img: %-10s error: %.4f time: %.4f' % (img_name, err, run_time))
    result['graph-iht'] = [err, run_time]
    # ----------------------- Graph-Sto-IHT --------------------------
    err, run_time = cv_graph_sto_iht(y0, y_va, omega_va,
                                     k, c, opts, x_gray, edges, costs, omega)
    print("GraphStoIHT -- Trial_%02d Oversampling factor:(%.1f, m=%04d)" %
          (trial_i, m_vec[m_ind], m)),
    print('img: %-10s error: %.4f time: %.4f' % (img_name, err, run_time))
    result['graph-sto-iht'] = [err, run_time]
    return trial_i, img_name, m_ind, result


def run_single_algo_2(para):
    np.random.seed()
    trial_i, img_name, m_vec, m_ind, p, k, c, opts, x_gray = para
    result = dict()
    m = int(2 * round(m_vec[m_ind] * k / 2))
    perm = np.arange(0, p)
    q = np.random.permutation(p / 2 - 1) + 1
    omega = q[:m / 2]
    m_f = lambda x: mat_f(x, omega, perm)  # Fourier operator.
    m_trans_f = lambda x: mat_trans_f(x, p, omega, perm)
    y0 = m_f(x_gray)
    y = y0
    # ---------------------- CoSaMP ----------------------------------
    re = algo_cosamp(y, k, c, m_f, m_trans_f, opts, x_gray)
    x_hat, time_cgs, time_model, time_proj, run_time = re
    err = norm(x_gray - x_hat) ** 2. / norm(x_gray) ** 2.
    print("CoSaMP      -- Trial_%02d Oversampling factor:(%.1f, m=%04d)" %
          (trial_i, m_vec[m_ind], m)),
    print('img: %-10s error: %.4f time: %.4f' % (img_name, err, run_time))
    result['cosamp'] = [err, run_time]
    # ---------------------- Graph-CoSaMP ----------------------------
    edges, costs = opts['edges'], opts['costs']
    re = algo_graph_cosamp(
        y, k, c, m_f, m_trans_f, opts, x_gray, edges, costs)
    x_hat, time_cgs, time_model, time_proj, run_time = re
    err = norm(x_gray - x_hat) ** 2. / norm(x_gray) ** 2.
    print("GraphCoSaMP -- Trial_%02d Oversampling factor:(%.1f, m=%04d)" %
          (trial_i, m_vec[m_ind], m)),
    print('img: %-10s error: %.4f time: %.4f' % (img_name, err, run_time))
    result['graph-cosamp'] = [err, run_time]
    # ----------------------- Graph-IHT --------------------------
    err, run_time = cv_graph_iht(k, c, opts, x_gray, edges, costs, omega)
    print("GraphIHT    -- Trial_%02d Oversampling factor:(%.1f, m=%04d)" %
          (trial_i, m_vec[m_ind], m)),
    print('img: %-10s error: %.4f time: %.4f' % (img_name, err, run_time))
    result['graph-iht'] = [err, run_time]
    # ----------------------- Graph-Sto-IHT --------------------------
    err, run_time = cv_graph_sto_iht(k, c, opts, x_gray, edges, costs, omega)
    print("GraphStoIHT -- Trial_%02d Oversampling factor:(%.1f, m=%04d)" %
          (trial_i, m_vec[m_ind], m)),
    print('img: %-10s error: %.4f time: %.4f' % (img_name, err, run_time))
    result['graph-sto-iht'] = [err, run_time]
    return trial_i, img_name, m_ind, result


def run_test(sample_ratio_arr, img_name_list, num_cpus, trial_range, root_p):
    import scipy.io as sio
    img_list = [sio.loadmat(root_p + 'image_%s.mat' % _)['x_gray']
                for _ in img_name_list]
    c_list = [3, 1, 4]
    edges, costs = simu_grid_graph(width=100, height=100)
    opts = {'select_atom': 2,
            'gamma': 0.1,
            'tol': 1e-4,
            'max_iter': 50,
            'verbose': 0,
            'tol_early': 5e-3,
            'debias': 0,
            'hhs': 0,
            'im_height': 100,
            'im_width': 100,
            'edges': edges,
            'costs': costs,
            'display_perf': 1,
            'c': {
                'background': 3,
                'angio': 1,
                'icml': 4}}
    tol = 5e-2
    input_paras = []
    for trial_i in trial_range:
        for img, img_name, c in zip(img_list, img_name_list, c_list):
            x_gray = img.flatten()
            p = len(x_gray)
            k = len(np.nonzero(x_gray)[0])
            for m_ind in range(len(sample_ratio_arr)):
                re = (trial_i, img_name, sample_ratio_arr,
                      m_ind, p, k, c, opts, x_gray)
                input_paras.append(re)
    pool = multiprocessing.Pool(processes=num_cpus)
    results_pool = pool.map(run_single_algo, input_paras)
    pool.close()
    pool.join()
    method_list = ['cosamp', 'graph-cosamp', 'graph-sto-iht', 'graph-iht']
    for trial_i in trial_range:
        all_results = {method: {img_name: np.zeros(len(sample_ratio_arr))
                                for img_name in img_name_list}
                       for method in method_list}
        for _, img_name, m_ind, re in results_pool:
            if _ == trial_i:
                for method in re:
                    val = 1.0 if re[method][0] < tol else 0.0
                    all_results[method][img_name][m_ind] = val
        f_name = root_p + 'sr_test08_trial_%03d.pkl' % trial_i
        pickle.dump(all_results, open(f_name, 'wb'))


def show_test(
        trial_range, img_name_list, sample_ratio_arr,
        method_list, method_label_list, root_p):
    sum_results = {img_name: {method: np.zeros((len(trial_range),
                                                len(sample_ratio_arr)))
                              for method in method_list} for img_name in
                   img_name_list}
    for trial_i in trial_range:
        f_name = root_p + 'sr_test08_trial_%03d.pkl' % trial_i
        print('load file from: %s' % f_name)
        results = pickle.load(open(f_name))
        for method in results:
            for img_name in results[method]:
                ind = list(trial_range).index(trial_i)
                sum_results[img_name][method][ind] = \
                    results[method][img_name]

    for img_name in sum_results:
        for method in sum_results[img_name]:
            re = sum_results[img_name][method]
            sum_results[img_name][method] = np.mean(re, axis=0)

    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    rc('text', usetex=True)

    img_data = get_img_data(root_p)  # 236, 383, 411
    resized_images = [img_data[_] for _ in img_data['img_list']]

    rcParams['figure.figsize'] = 8, 5
    color_list = ['c', 'b', 'r', 'k', 'm', 'y', 'r']
    marker_list = ['X', 'o', 'D', 'h', 'P', 'p', 's']
    img_name_list = ['background', 'angio', 'icml']
    title_list = ['BackGround', 'Angio', 'Text']

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    plt.figure(figsize=(8, 5))
    grid = gridspec.GridSpec(2, 15)
    ax00 = plt.subplot(grid[0, 0:4])
    plt.xticks(())
    plt.yticks(())
    ax01 = plt.subplot(grid[0, 4:8])
    plt.xticks(())
    plt.yticks(())
    ax02 = plt.subplot(grid[0, 8:12])
    plt.xticks(())
    plt.yticks(())
    ax03 = plt.subplot(grid[0, 12:15])
    plt.xticks(())
    plt.yticks(())
    ax10 = plt.subplot(grid[1, 0:5])
    ax11 = plt.subplot(grid[1, 5:10])
    ax12 = plt.subplot(grid[1, 10:15])
    ax = np.asarray([[ax00, ax01, ax02], [ax10, ax11, ax12]])
    for img_ind, img_name in enumerate(img_name_list):
        ax[1, img_ind].grid(b=True, linestyle='dotted', c='lightgray')
        ax[1, img_ind].set_xticks([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        ax[1, img_ind].set_yticks(np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        for method_ind, method in enumerate(method_list):
            ax[1, img_ind].plot(
                sample_ratio_arr, sum_results[img_name][method],
                c=color_list[method_ind],
                markerfacecolor='none', linestyle='-',
                marker=marker_list[method_ind], markersize=5.,
                markeredgewidth=1.0, linewidth=1.0,
                label=method_label_list[method_ind])
        ax[1, img_ind].set_xlabel('Oversampling ratio $\displaystyle n / s $',
                                  labelpad=0)
    ax[1, 0].set_ylabel('Probability of Recovery', labelpad=0)
    for i in range(3):
        ax[0, i].set_title(title_list[i])
        ax[0, i].imshow(np.reshape(resized_images[i], (100, 100)), cmap='gray')
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
    ax03.plot()
    ax03.set_xticks([])
    ax03.set_yticks([])
    for str_ in ['right', 'top', 'left']:
        ax03.spines[str_].set_visible(False)
    plt.setp(ax[1, 1].get_yticklabels(), visible=False)
    plt.setp(ax[1, 2].get_yticklabels(), visible=False)
    ax[1, 2].legend(loc='center right', fontsize=10,
                    bbox_to_anchor=(1.05, 1.35),
                    frameon=True, borderpad=0.1, labelspacing=0.2,
                    handletextpad=0.1, markerfirst=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.0)
    f_name = root_p + 'sr_simu_test08.pdf'
    print('save fig to: %s' % f_name)
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0,
                format='pdf')
    plt.close()


def get_img_data(root_p):
    import scipy.io as sio
    img_name_list = ['background', 'angio', 'icml']
    g_list = [3, 1, 4]
    height, width = 100, 100
    data = dict()
    s_list = []
    for img_ind, _ in enumerate(img_name_list):
        img = sio.loadmat(root_p + 'image_%s.mat' % _)['x_gray']
        im = np.asarray(img).reshape((height, width))
        data[_] = im
        s_list.append(len(np.nonzero(data[_])[0]))
    img_data = {
        'img_list': img_name_list,
        'background': np.asarray(data['background']).flatten(),
        'angio': np.asarray(data['angio']).flatten(),
        'icml': np.asarray(data['icml']).flatten(),
        'height': height,  # grid size (length).
        'width': width,  # grid width (width).
        'p': height * width,
        's': {_: s_list[ind] for ind, _ in enumerate(img_name_list)},
        'g': {_: g_list[ind] for ind, _ in enumerate(img_name_list)},
        's_list': s_list,  # sparsity list
        'graph': simu_grid_graph(height=height, width=width)
    }
    return img_data


def show_experiment1(root):
    import scipy.io as sio
    img_name_list = ['background', 'angio', 'icml']
    img_list = [sio.loadmat(root + '%s.mat' % _)['x_gray']
                for _ in img_name_list]
    results = pickle.load(open(root + 'output/test_results.pkl'))
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rc('font', **{'size': 15})
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 15, 8
    m_vec = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4,
             4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0]
    color_list = ['r', 'b', 'r', 'c', 'm', 'y']
    marker_list = ['x', 'o', '*', '.', 'x', 's']
    face_color_list = ['none', 'none', 'r', 'c', 'm', 'none']
    method_list = ['cosamp', 'graph_cosamp']
    method_label_list = ['CoSaMP', 'GraphCoSaMP']
    img_name_list = ['background', 'angio', 'icml']
    title_list = ['BackGround', 'Angio', 'Text']
    fig, ax = plt.subplots(2, 3)
    for img_ind, img_name in enumerate(img_name_list):
        ax[1, img_ind].grid(b=True, linestyle='--')
        ax[1, img_ind].set_xticks([2, 3, 4, 5, 6, 7])
        ax[1, img_ind].set_yticks(np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        for method_ind, method in enumerate(method_list):
            ax[1, img_ind].plot(
                m_vec, np.mean(results[img_name]['error_%s' % method], axis=0),
                c=color_list[method_ind],
                markerfacecolor=face_color_list[method_ind], linestyle='-',
                marker=marker_list[method_ind], markersize=10.,
                markeredgewidth=2., linewidth=2.0,
                label=method_label_list[method_ind])
            print(np.mean(results[img_name]['error_%s' % method]))
        ax[1, img_ind].set_ylabel('Probability of Recovery')
        ax[1, img_ind].set_xlabel('Oversampling ratio $\displaystyle n / s $')
    for i in range(3):
        ax[0, i].set_title(title_list[i])
        ax[0, i].imshow(img_list[i])
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
    ax[1, 0].legend(loc='lower right')
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    if not os.path.exists(root + 'figs'):
        os.mkdir(root + 'figs')
    f_name = root + 'figs/img_text_icml15.pdf'
    print('save fig to: %s' % f_name)
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0,
                format='pdf')
    plt.close()


def main():
    sample_ratio_arr = np.arange(start=2., stop=7.1, step=0.2)
    method_list = ['cosamp', 'graph-cosamp', 'graph-sto-iht', 'graph-iht']
    method_label_list = ['CoSaMP', 'GraphCoSaMP', 'GraphStoIHT', 'GraphIHT']
    command = sys.argv[1]
    root_p = '/network/rit/lab/ceashpc/bz383376/data/icml19/publish/'
    img_name_list = ['background', 'angio', 'icml']
    if command == 'run_test':
        num_cpus = int(sys.argv[2])
        trial_range = range(int(sys.argv[3]), int(sys.argv[4]))
        run_test(sample_ratio_arr=sample_ratio_arr,
                 img_name_list=img_name_list,
                 num_cpus=num_cpus,
                 trial_range=trial_range,
                 root_p=root_p)
    elif command == 'show_test':
        trial_range = range(20)
        show_test(
            trial_range=trial_range,
            img_name_list=img_name_list,
            sample_ratio_arr=sample_ratio_arr,
            method_list=method_list,
            method_label_list=method_label_list,
            root_p=root_p)


if __name__ == '__main__':
    main()
