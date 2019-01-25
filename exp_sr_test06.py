# -*- coding: utf-8 -*-
"""
TODO: Please check readme.txt file first!
--
This Python2.7 program is to reproduce Figure-5. In this test, we compare
GraphStoIHT with six baseline methods on the real image dataset, which can be
found in reference [2].

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
    [7] Blumensath, Thomas, and Mike E. Davies. "Normalized iterative hard
        thresholding: Guaranteed stability and performance." IEEE Journal
        of selected topics in signal processing 4.2 (2010): 298-309.

# TODO You need to:
    1.  install numpy, matplotlib (optional), and networkx (optional).
    2.  build our sparse_module by executing ./build.sh please check our
        readme.md file. If you do not know how to compile this library.
"""
import os
import time
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

np.random.seed()
g_x_tr_mat = np.random.normal(0.0, 1.0, 2500 * 2500)
g_x_va_mat = np.random.normal(0.0, 1.0, 100 * 2500)


def print_helper(method, trial_i, n, err, num_epochs, run_time):
    print('%-13s trial_%03d n: %03d w_error: %.3e '
          'num_epochs: %03d run_time: %.3e' %
          (method, trial_i, n, err, num_epochs, run_time))


def get_img_data(root_p):
    import scipy.io as sio
    from PIL import Image
    img_name_list = ['background', 'angio', 'icml']
    re_height, re_width = 50, 50
    resized_data = dict()
    s_list = []
    for img_ind, _ in enumerate(img_name_list):
        img = sio.loadmat(root_p + 'sr_image_%s.mat' % _)['x_gray']
        im = Image.fromarray(img).resize((re_height, re_width), Image.BILINEAR)
        im = np.asarray(im.getdata()).reshape((re_height, re_width))
        resized_data[_] = im
        s_list.append(len(np.nonzero(resized_data[_])[0]))
    img_data = {
        'img_list': img_name_list,
        'background': np.asarray(resized_data['background']).flatten(),
        'angio': np.asarray(resized_data['angio']).flatten(),
        'icml': np.asarray(resized_data['icml']).flatten(),
        'height': re_height,
        'width': re_width,
        'p': re_height * re_width,
        's': {_: s_list[ind] for ind, _ in enumerate(img_name_list)},
        's_list': s_list,
        'g_dict': {'background': 1, 'angio': 1, 'icml': 4},
        'graph': simu_grid_graph(height=re_height, width=re_width)
    }
    return img_data


def simu_grid_graph(width, height):
    """ Generate a grid graph with size, width x height. Totally there will be
        width x height number of nodes in this generated graph.
    :param width:       the width of the grid graph.
    :param height:      the height of the grid graph.
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
    weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights


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
    :return:            1.  the number of epochs(iterations) used,
                        2.  the run time.
                        3.  the final estimator,
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
    """ Stochastic Iterative Hard Thresholding Method proposed in [1].
    :param x_mat:       the design matrix.
    :param y_tr:        the array of measurements.
    :param max_epochs:  the maximum epochs (iterations) allowed.
    :param lr:          the learning rate (should be 1.0).
    :param s:           the sparsity parameter.
    :param x0:          x0 is the initial point.
    :param tol_algo:    tolerance parameter for early stopping.
    :param b:           block size
    :return:            1.  the number of epochs(iterations) used,
                        2.  the run time.
                        3.  the final estimator,
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
        x_mat, y_tr, max_epochs, lr, x0, tol_algo, edges, costs, g, s,
        root=-1, gamma=0.1, proj_max_num_iter=50, verbose=0):
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
                 x0, tol_algo, edges, costs, g, s):
    """ Tuning parameter by using additional validation dataset. """
    test_err_mat = np.zeros(len(lr_list))
    x_hat_dict = dict()
    for lr_ind, lr in enumerate(lr_list):
        num_epochs, run_time, x_hat = algo_graph_iht(
            x_mat=x_tr_mat, y_tr=y_tr, max_epochs=max_epochs, lr=lr, x0=x0,
            tol_algo=tol_algo, edges=edges, costs=costs, g=g, s=s)
        y_err = np.linalg.norm(y_va - np.dot(x_va_mat, x_hat)) ** 2.
        test_err_mat[lr_ind] = y_err
        x_hat_dict[lr] = (num_epochs, run_time, x_hat)
    min_index = np.argmin(test_err_mat)
    best_lr = lr_list[min_index]
    err = np.linalg.norm(x_star - x_hat_dict[best_lr][2])
    num_epochs, run_time = x_hat_dict[best_lr][:2]
    return err, num_epochs, run_time


def algo_graph_sto_iht(
        x_mat, y_tr, max_epochs, lr, x0, tol_algo, edges, costs, g, s, b,
        root=-1, gamma=0.1, proj_max_num_iter=50, verbose=0):
    np.random.seed()
    start_time = time.time()
    x_hat = np.copy(x0)
    (n, p) = x_mat.shape
    x_tr_t = np.transpose(x_mat)
    b = n if n < b else b
    num_blocks = int(n) / int(b)
    prob = [1. / num_blocks] * num_blocks

    # graph projection para
    h_low = int(len(x0) / 2)
    h_high = int(h_low * (1. + gamma))
    t_low = int(s)
    t_high = int(s * (1. + gamma))

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
                     max_epochs, tol_algo, x_star, x0, edges, costs, g, s):
    """ Tuning parameter by using additional validation dataset. """
    test_err_mat = np.zeros(len(lr_list) * len(b_list))
    para_dict = dict()
    x_hat_dict = dict()
    for index, (lr, b) in enumerate(product(lr_list, b_list)):
        num_epochs, run_time, x_hat = algo_graph_sto_iht(
            x_mat=x_tr_mat, y_tr=y_tr, max_epochs=max_epochs,
            lr=lr, x0=x0, tol_algo=tol_algo, edges=edges,
            costs=costs, g=g, s=s, b=b)
        y_err = np.linalg.norm(y_va - np.dot(x_va_mat, x_hat)) ** 2.
        test_err_mat[index] = y_err
        para_dict[index] = (lr, b)
        x_hat_dict[(lr, b)] = (num_epochs, run_time, x_hat)
    lr, b = para_dict[int(np.argmin(test_err_mat))]
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
        h_g, t_g, s, root=-1, gamma=0.1, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    x_hat = np.zeros_like(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y_tr)
    num_epochs = 0

    h_low, h_high = int(2 * s), int(2 * s * (1.0 + gamma))
    t_low, t_high = int(s), int(s * (1.0 + gamma))

    for epoch_i in range(max_epochs):
        num_epochs += 1
        grad = -2. * (np.dot(xtx, x_hat) - xty)  # proxy
        head_nodes, proj_grad = algo_head_tail_bisearch(
            edges, grad, costs, h_g, root,
            h_low, h_high, proj_max_num_iter, verbose)
        gamma = np.union1d(x_hat.nonzero()[0], head_nodes)
        bt = np.zeros_like(x_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_mat[:, gamma]), y_tr)
        tail_nodes, proj_bt = algo_head_tail_bisearch(
            edges, bt, costs, t_g, root,
            t_low, t_high, proj_max_num_iter, verbose)
        x_hat = proj_bt

        if np.linalg.norm(x_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break

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


def show_resized_figures(root_p, re_height=50, re_width=50):
    np.random.seed()
    import scipy.io as sio
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 15
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 8, 5

    fig, ax = plt.subplots(2, 3)
    title_list = ['BackGround', 'Angiogram', 'Text']
    for img_ind, img_name in enumerate(['background', 'angio', 'icml']):
        img = sio.loadmat(root_p + 'sr_image_%s.mat' % img_name)['x_gray']
        im = Image.fromarray(img).resize((re_height, re_width), Image.BILINEAR)
        im = np.asarray(im.getdata()).reshape((re_height, re_width))
        ax[0, img_ind].imshow(img, cmap='gray')
        ax[0, img_ind].set_title(title_list[img_ind])
        ax[0, img_ind].set_xticks([20, 40, 60, 80])
        ax[0, img_ind].set_yticks([20, 40, 60, 80])
        ax[1, img_ind].set_xticks([10, 20, 30, 40])
        ax[1, img_ind].set_yticks([10, 20, 30, 40])
        ax[1, img_ind].imshow(im, cmap='gray')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    f_name = root_p + 'images_resized_50_50.pdf'
    print('save fig to: %s' % f_name)
    plt.savefig(f_name, dpi=600, bbox_inches='tight',
                pad_inches=0, format='pdf')
    plt.close()


def run_single_test(data):
    method = data['method']
    img_name = data['img_name']
    trial_i = data['trial_i']
    n = data['n']
    p = data['p']
    x_star = data['x_star']
    max_epochs = data['max_epochs']
    lr_list = data['lr_list']
    b_list = data['b_list']
    s = data['s']
    x0 = data['x0']
    tol_algo = data['tol_algo']
    x_tr_mat_ = np.reshape(g_x_tr_mat[:n * p], (n, p)) / np.sqrt(n)
    x_va_mat_ = np.reshape(g_x_tr_mat[:100 * p], (100, p)) / np.sqrt(100.)
    y_tr = np.dot(x_tr_mat_, x_star)
    y_va = np.dot(x_va_mat_, x_star)

    edges = data['proj_para']['edges']
    costs = data['proj_para']['costs']
    g = data['proj_para']['g']
    if method == 'niht':
        err, num_epochs, run_time = algo_niht(
            x_tr_mat_, y_tr, max_epochs, s, x_star, x0, tol_algo)
    elif method == 'iht':
        err, num_epochs, run_time = cv_iht(
            x_tr_mat_, y_tr, x_va_mat_, y_va, max_epochs,
            lr_list, s, x_star, x0, tol_algo)
    elif method == 'sto-iht':
        err, num_epochs, run_time = cv_sto_iht(
            x_tr_mat_, y_tr, x_va_mat_, y_va, max_epochs, s, x_star, x0,
            tol_algo, b_list, lr_list)
    elif method == 'graph-iht':
        err, num_epochs, run_time = cv_graph_iht(
            x_tr_mat_, y_tr, x_va_mat_, y_va, max_epochs, lr_list, x_star,
            x0, tol_algo, edges, costs, g, s)
    elif method == 'graph-sto-iht':
        err, num_epochs, run_time = cv_graph_sto_iht(
            x_tr_mat_, y_tr, x_va_mat_, y_va, b_list, lr_list, max_epochs,
            tol_algo, x_star, x0, edges, costs, g, s)
    elif method == 'graph-cosamp':
        err, num_epochs, run_time = algo_graph_cosamp(
            x_tr_mat_, y_tr, max_epochs, x_star, x0, tol_algo, edges, costs,
            h_g=int(2.0 * g), t_g=g, s=s)
    elif method == 'cosamp':
        err, num_epochs, run_time = algo_cosamp(
            x_tr_mat_, y_tr, max_epochs, x_star, x0, tol_algo, s)
    else:
        print('something must wrong.')
        exit()
        err, num_epochs, run_time = 0.0, 0.0, 0.0
    print_helper(method, trial_i, n, err, num_epochs, run_time)
    return method, img_name, trial_i, n, err


def run_test(trial_range, max_epochs, tol_algo, tol_rec,
             sample_ratio_arr, method_list, num_cpus, root_input, root_output):
    """ This test is test the methods on 50x50 resized images. """
    np.random.seed()
    start_time = time.time()
    img_data = get_img_data(root_input)  # 236, 383, 411
    edges, costs = img_data['graph']
    input_data_list = []
    for img_name in img_data['img_list']:
        p = img_data['p']
        s = img_data['s'][img_name]
        g = img_data['g_dict'][img_name]
        x_star = img_data[img_name]
        n_list = [int(_ * s) for _ in sample_ratio_arr]
        for trial_i in trial_range:
            for n in n_list:
                for method in method_list:
                    data = {
                        'trial_i': trial_i,
                        's': s,
                        'n': n,
                        'p': p,
                        'img_name': img_name,
                        'max_epochs': max_epochs,
                        'n_list': n_list,
                        'lr_list': [0.2, 0.4, 0.6, 0.8],
                        'b_list': [int(n) / 5, int(n) / 10],
                        'x_star': x_star,
                        'x0': np.zeros(p),
                        'subgraph': np.nonzero(x_star)[0],
                        'tol_algo': tol_algo,
                        'height': img_data['height'],
                        'width': img_data['width'],
                        'tol_rec': tol_rec,
                        'img_data': img_data,
                        'verbose': 0,
                        'method': method,
                        'proj_para': {'edges': edges, 'costs': costs, 'g': g}
                    }
                    input_data_list.append(data)
    pool = multiprocessing.Pool(processes=int(num_cpus))
    results_pool = pool.map(run_single_test, input_data_list)
    pool.close()
    pool.join()

    sum_results = dict()  # trial_i, n, rec_err
    for method, img_name, trial_i, n, err in results_pool:
        if trial_i not in sum_results:
            sum_results[trial_i] = []
        sum_results[trial_i].append((method, img_name, trial_i, n, err))
    for trial_i in sum_results:
        f_name = root_output + 'results_exp_sr_test06_trial_%02d.pkl' % trial_i
        print('save results to file: %s' % f_name)
        pickle.dump({'results_pool': sum_results[trial_i]},
                    open(f_name, 'wb'))
    print('total run time of %02d trials: %.2f seconds.' %
          (len(trial_range), time.time() - start_time))


def show_test(method_list, method_label_list, sample_ratio_arr, root_p):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    rc('text', usetex=True)

    img_data = get_img_data('data/')  # 236, 383, 411
    resized_images = [img_data[_] for _ in img_data['img_list']]

    rcParams['figure.figsize'] = 8, 5
    f_name = root_p + 'results_exp_sr_test06.pkl'
    trim_results = pickle.load(open(f_name))['trim_results']
    color_list = ['c', 'b', 'g', 'k', 'm', 'y', 'r']
    marker_list = ['D', 'X', 'o', 'h', 'P', 'p', 's']
    img_name_list = ['background', 'angio', 'icml']
    title_list = ['(a) Background', '(b) Angio', '(c) Text']

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
        ax[1, img_ind].set_xticks([1.5, 2.0, 2.5, 3.0, 3.5])
        ax[1, img_ind].set_yticks(np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        for method_ind, method in enumerate(method_list):
            ax[1, img_ind].plot(
                sample_ratio_arr, trim_results[img_name][method],
                c=color_list[method_ind],
                markerfacecolor='none', linestyle='-',
                marker=marker_list[method_ind], markersize=6.,
                markeredgewidth=1.0, linewidth=1.0,
                label=method_label_list[method_ind])
        ax[1, img_ind].set_xlabel('Oversampling ratio $\displaystyle m / s $',
                                  labelpad=0)
    ax[1, 0].set_ylabel('Probability of Recovery', labelpad=0)
    for i in range(3):
        ax[0, i].set_title(title_list[i])
        ax[0, i].imshow(np.reshape(resized_images[i], (50, 50)), cmap='gray')
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
    f_name = root_p + 'results_exp_sr_test06.pdf'
    print('save fig to: %s' % f_name)
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0,
                format='pdf')
    plt.close()


def summarize_results(
        trial_range, sample_ratio_arr, method_list, tol_rec,
        trim_ratio, root_p):
    results_pool = []
    num_trials = len(trial_range)
    for trial_i in trial_range:
        f_name = root_p + 'results_exp_sr_test06_trial_%02d.pkl' % trial_i
        print('load file from: %s' % f_name)
        results = pickle.load(open(f_name))
        for item in results:
            results_pool.extend(results[item])
    img_data = get_img_data('data/')
    sum_results = dict()
    for method, fig_i, trial_i, n, err in results_pool:
        print(method, err)
        n_list = [int(_ * img_data['s'][fig_i]) for _ in sample_ratio_arr]
        n_ind = list(n_list).index(n)
        trial_i_ind = list(trial_range).index(trial_i)
        if fig_i not in sum_results:
            sum_results[fig_i] = dict()
        if method not in sum_results[fig_i]:
            sum_results[fig_i][method] = np.zeros((num_trials, len(n_list)))
        sum_results[fig_i][method][trial_i_ind][n_ind] = err
    # trim 5% of the results (rounding when necessary).
    num_trim = int(round(trim_ratio * num_trials))
    trim_results = {
        fig_i: {
            method: np.zeros(
                (num_trials - 2 * num_trim, len(sample_ratio_arr)))
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
    f_name = root_p + 'results_exp_sr_test06.pkl'
    print('save file to: %s' % f_name)
    pickle.dump({'results_pool': results_pool,
                 'trim_results': trim_results,
                 'sum_results': sum_results}, open(f_name, 'wb'))


def main():
    num_trials = 50
    sample_ratio_arr = np.arange(start=1.5, stop=3.6, step=0.1)

    max_epochs = 500
    tol_algo = 1e-7
    tol_rec = 1e-6
    # the trimmed ratio
    # ( about 5% of the best and worst have been removed).
    trim_ratio = 0.05
    method_list = ['niht', 'iht', 'sto-iht', 'cosamp',
                   'graph-iht', 'graph-cosamp', 'graph-sto-iht']

    # TODO config the path by yourself.
    root_p = 'results/'
    if not os.path.exists(root_p):
        os.mkdir(root_p)

    if len(os.sys.argv) <= 1:
        print('\n'.join(['please use one of the following commands: ',
                         '1. python exp_sr_test06.py run_test 50 0 10',
                         '2. python exp_sr_test06.py show_test']))
        exit(0)

    command = os.sys.argv[1]
    if command == 'run_test':
        num_cpus = int(os.sys.argv[2])
        trial_range = range(int(os.sys.argv[3]), int(os.sys.argv[4]))
        for trial_i in trial_range:
            np.random.seed()
            run_test(trial_range=[trial_i],
                     max_epochs=max_epochs,
                     tol_algo=tol_algo,
                     tol_rec=tol_rec,
                     sample_ratio_arr=sample_ratio_arr,
                     method_list=method_list,
                     num_cpus=num_cpus,
                     root_input='data/', root_output='results/')
    elif command == 'summarize_results':
        trial_range = range(num_trials)
        summarize_results(
            trial_range=trial_range,
            sample_ratio_arr=sample_ratio_arr,
            method_list=method_list,
            tol_rec=tol_rec,
            trim_ratio=trim_ratio,
            root_p=root_p)
    elif command == 'show_test':
        method_label_list = ['NIHT', 'IHT', 'StoIHT', 'CoSaMP', 'GraphIHT',
                             'GraphCoSaMP', 'GraphStoIHT']
        show_test(method_list=method_list,
                  method_label_list=method_label_list,
                  sample_ratio_arr=sample_ratio_arr,
                  root_p=root_p)
    elif command == 'show_resized_figures':
        show_resized_figures(root_p=root_p)


if __name__ == '__main__':
    main()
