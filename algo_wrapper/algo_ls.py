# -*- coding: utf-8 -*-
import time
import numpy as np
from itertools import product

__all__ = ['algo_iht',
           'cv_iht',
           'cv_sto_iht',
           'cv_graph_iht',
           'cv_graph_sto_iht',
           'algo_sto_iht',
           'algo_graph_iht',
           'algo_graph_sto_iht',
           'algo_niht',
           'algo_cosamp',
           'algo_graph_cosamp',
           'algo_best_subset',
           'algo_head_tail_binsearch']

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


def _fold_split(sample_indices, k, random_perm=False):
    """To split the training samples into k folders.
    :param sample_indices:
    :param k: split it into k folder.
    :param random_perm:True to shuffle the samples.
    :return:
    """
    if random_perm:
        shuffle_indices = np.random.permutation(len(sample_indices))
        sample_indices = sample_indices[shuffle_indices]
    step_size = int(len(sample_indices) / k)
    k_fold_dict = dict()
    for fold_i in range(k):
        start = fold_i * step_size
        end = (fold_i + 1) * step_size
        if fold_i == k - 1:
            k_fold_dict[fold_i] = sample_indices[start:]
        else:
            k_fold_dict[fold_i] = sample_indices[start:end]
    re_folding = []
    for ii, fold_ind in enumerate(range(k)):
        test_fold = k_fold_dict[fold_ind]
        train_fold = [kk for _ in k_fold_dict
                      for kk in k_fold_dict[_] if _ != fold_ind]
        re_folding.append((ii, train_fold, test_fold))
    return re_folding


def algo_head_tail_binsearch(
        edges, x, costs, g, root, s_low, s_high, max_num_iter, verbose):
    """Head and Tail projection wrapper.
    :param edges: list of edges.
    :param x: projection input vector.
    :param costs: costs of edges.
    :param g: number of connected component.
    :param root: root of the tree. -1: non-root.
    :param s_low: lower bound of the sparsity.
    :param s_high: upper bound of the sparsity.
    :param max_num_iter: maximum number of iterations allowed.
    :param verbose: print out some information.
    :return:
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


def algo_iht(x_mat, y_tr, max_epochs, lr, s, x_star, x0, tol_algo, verbose,
             save_epoch=False):
    """Iterative Hard Thresholding Method proposed in [1].
    The standard iterative hard thresholding method for compressive sensing.
    Reference:  [1] Blumensath, Thomas, and Mike E. Davies.
                    "Iterative hard thresholding for compressed sensing."
                    Applied and computational harmonic analysis 27.3
                    (2009): 265-274.
    :param x_mat: the design matrix.
    :param y_tr: the array of measurements.
    :param max_epochs: the maximum epochs (iterations) allowed.
    :param lr: the learning rate should be 1.0 consistent with the paper.
    :param s: the sparsity parameter.
    :param x_star: x_star is the true signal.
    :param x0: x0 is the initial point.
    :param tol_algo: tolerance parameter for early stopping.
    :param verbose: print out some info.
    :param save_epoch: True to save the results during the training.
    :return: list of recovery error during training.
    """
    start_time = time.time()
    x_hat = x0
    (n, p) = x_mat.shape
    x_tr_t = np.transpose(x_mat)
    xtx = np.dot(x_tr_t, x_mat)
    xty = np.dot(x_tr_t, y_tr)

    num_epochs = 0
    y_err_list = []
    x_err_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        bt = x_hat - lr * (np.dot(xtx, x_hat) - xty)
        bt[np.argsort(np.abs(bt))[0:p - s]] = 0.  # thresholding step
        x_hat = bt

        # ------------------
        if save_epoch:
            y_err = np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) ** 2.
            y_err_list.append(y_err)
        if x_star is not None:
            x_err_list.append(np.linalg.norm(x_hat - x_star))
        if verbose > 0:
            print('epoch_%03d x_err: %.6e y_err: %.6e' %
                  (epoch_i, x_err_list[-1], y_err_list[-1]))

        # early stopping for diverge cases due to the large learning rate
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
        # early stopping for the local optimal point.
        # algorithm will never improve the solution.
        if epoch_i >= 10 and \
                len(np.unique(y_err_list[-5:])) == 1 and \
                len(np.unique(x_err_list[-5:])) == 1:
            break
    run_time = time.time() - start_time
    if x_star is not None:
        return x_err_list, y_err_list, \
               num_epochs, run_time, x_err_list[-1], x_hat
    else:
        return x_err_list, y_err_list, \
               num_epochs, run_time, -1.0, x_hat


def cv_iht(x_mat, y_tr, max_epochs, lr_list, s, x_star, x0, tol_algo, verbose,
           num_fold):
    """Use 10 folder cross validation to select learning rate """
    re_folding = _fold_split(sample_indices=range(len(y_tr)), k=num_fold)
    test_err_mat = np.zeros(shape=(len(lr_list), num_fold))
    for lr_ind, lr in enumerate(lr_list):
        for ii, tr, te in re_folding:
            _, _, num_epochs_, run_time_, _, x_hat = algo_iht(
                x_mat=x_mat[tr], y_tr=y_tr[tr], max_epochs=max_epochs,
                lr=lr, s=s, x_star=None, x0=x0, tol_algo=tol_algo,
                verbose=verbose)
            err = np.linalg.norm(y_tr[te] - np.dot(x_mat[te], x_hat)) ** 2.
            test_err_mat[lr_ind][ii] += err
    min_index = np.argmin(np.mean(test_err_mat, axis=1))
    best_lr = lr_list[min_index]
    re = algo_iht(
        x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs, lr=best_lr,
        s=s, x_star=x_star, x0=x0, tol_algo=tol_algo, verbose=verbose)
    return best_lr, re


def algo_sto_iht(
        x_mat, y_tr, max_epochs, lr, s, x_star, x0, tol_algo, verbose, b,
        save_iter=False):
    """Stochastic Iterative Hard Thresholding Method proposed in [1]
    Reference:  [1] Nguyen, Nam, Deanna Needell, and Tina Woolf.
                    "Linear convergence of stochastic iterative greedy
                    algorithms with sparse constraints." IEEE Transactions
                    on Information Theory 63.11 (2017): 6869-6895.
    :param x_mat: the design matrix.
    :param y_tr: the array of measurements.
    :param max_epochs: the maximum epochs (iterations) allowed.
    :param lr: the learning rate should be 1.0 consistent with the paper.
    :param s: the sparsity parameter.
    :param x_star: x_star is the true signal.
    :param x0: x0 is the initial point.
    :param tol_algo: tolerance parameter for early stopping.
    :param verbose: print out some info.
    :param b: block size
    :param save_iter: to save iteration information
    :return: list of recovery error during training.
    """
    # do not forget the random seed.
    np.random.seed()
    start_time = time.time()
    x_hat = x0
    (n, p) = x_mat.shape
    x_tr_t = np.transpose(x_mat)
    # if the block size is too large. just use a single block.
    b = n if n < b else b
    num_blocks = int(n) / int(b)
    prob = [1. / num_blocks] * num_blocks

    num_epochs = 0
    y_err_list = []
    x_err_list = []
    x_err_iter_list = []
    y_err_iter_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        for _ in range(num_blocks):
            # without replacement.
            ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            xtx = np.dot(x_tr_t[:, block], x_mat[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            gradient = - 2. * (xty - np.dot(xtx, x_hat))
            bt = x_hat - (lr / (prob[ii] * num_blocks)) * gradient
            bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
            x_hat = bt

            if save_iter:
                if x_star is not None:
                    x_err = np.linalg.norm(x_hat - x_star)
                    x_err_iter_list.append(x_err)
                y_err = np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) ** 2.
                y_err_iter_list.append(y_err)

        y_err_list.append(np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) ** 2.)
        if x_star is not None:
            x_err_list.append(np.linalg.norm(x_hat - x_star))

        if verbose > 0:
            print('epoch_%03d x_err: %.6e y_err: %.6e' %
                  (epoch_i, x_err_list[-1], y_err_list[-1]))

        # early stopping for diverge cases due to the large learning rate
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
        # early stopping for the local optimal point.
        # algorithm will never improve the solution.
        if epoch_i >= 10 and \
                len(np.unique(y_err_list[-5:])) == 1 and \
                len(np.unique(x_err_list[-5:])) == 1:
            break

    run_time = time.time() - start_time
    if x_star is not None:
        return x_err_list, x_err_iter_list, y_err_list, y_err_iter_list, \
               num_epochs, run_time, x_err_list[-1], x_hat
    else:
        return x_err_list, x_err_iter_list, y_err_list, y_err_iter_list, \
               num_epochs, run_time, -1.0, x_hat


def cv_sto_iht(x_mat, y_tr, max_epochs, s, x_star, x0,
               tol_algo, verbose, b_list, lr_list, num_fold):
    """ Use 10 folder cross validation to select learning rate """
    re_folding = _fold_split(sample_indices=range(len(y_tr)), k=num_fold)
    test_err_mat = np.zeros(shape=(len(lr_list) * len(b_list), num_fold))
    para_dict, index = dict(), 0
    for lr, b in product(lr_list, b_list):
        for ii, tr, te in re_folding:
            _, _, _, _, num_epochs_, run_time_, _, x_hat = algo_sto_iht(
                x_mat=x_mat[tr], y_tr=y_tr[tr], max_epochs=max_epochs,
                lr=lr, s=s, x_star=None, x0=x0, tol_algo=tol_algo,
                verbose=verbose, b=b)
            err = np.linalg.norm(y_tr[te] - np.dot(x_mat[te], x_hat)) ** 2.
            test_err_mat[index][ii] += err
        para_dict[index] = (lr, b)
        index += 1
    min_index = np.argmin(np.mean(test_err_mat, axis=1))
    lr, b = para_dict[min_index]
    re = algo_sto_iht(
        x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs, lr=lr, s=s,
        x_star=x_star, x0=x0, tol_algo=tol_algo, verbose=verbose, b=b)
    return lr, b, re


def algo_graph_iht(
        x_mat, y_tr, max_epochs, lr, x_star, x0, tol_algo, edges, costs, g,
        root, h_low, h_high, t_low, t_high, proj_max_num_iter, verbose):
    """Graph Iterative Hard Thresholding.
    References [1]  Hegde, Chinmay, Piotr Indyk, and Ludwig Schmidt. "
                    Approximation Algorithms for Model-Based Compressive
                    Sensing." IEEE Trans. Information Theory 61.9
                    (2015): 5129-5147.
    :param x_mat: the design matrix.
    :param y_tr: the array of measurements.
    :param max_epochs: the maximum epochs (iterations) allowed.
    :param lr: the learning rate should be 1.0 consistent with the paper.
    :param x_star: x_star is the true signal.
    :param x0: x0 is the initial point.
    :param tol_algo: tolerance parameter for early stopping.
    :param edges: edges in the graph.
    :param costs: edge costs
    :param g: number of connected component in the true signal.
    :param root: the root included in the result.
    :param h_low: lower bound of sparsity of head projection.
    :param h_high: upper bound of sparsity of head projection.
    :param t_low: lower bound of sparsity of tail projection.
    :param t_high: upper bound of sparsity of tail projection.
    :param proj_max_num_iter: maximum number of iterations of projection.
    :param verbose: print out some information.
    :return:
    """
    start_time = time.time()
    x_hat, num_iter = x0, 0
    x_tr_t = np.transpose(x_mat)
    xtx = np.dot(x_tr_t, x_mat)
    xty = np.dot(x_tr_t, y_tr)

    num_epochs = 0
    y_err_list = []
    x_err_list = []
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

        y_err_list.append(np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) ** 2.)
        if x_star is not None:
            x_err_list.append(np.linalg.norm(x_hat - x_star))

        if verbose > 0:
            print('epoch_%03d x_err: %.6e y_err: %.6e' %
                  (epoch_i, x_err_list[-1], y_err_list[-1]))

        # early stopping for diverge cases due to the large learning rate
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
        # early stopping for the local optimal point.
        # algorithm will never improve the solution.
        if epoch_i >= 10 and \
                len(np.unique(y_err_list[-5:])) == 1 and \
                len(np.unique(x_err_list[-5:])) == 1:
            break
    run_time = time.time() - start_time
    if x_star is not None:
        return x_err_list, y_err_list, \
               num_epochs, run_time, x_err_list[-1], x_hat
    else:
        return x_err_list, y_err_list, \
               num_epochs, run_time, -1.0, x_hat


def cv_graph_iht(x_mat, y_tr, max_epochs, lr_list, x_star, x0, tol_algo,
                 edges, costs, g, root, s, h_factor, gamma, proj_max_num_iter,
                 verbose, num_fold):
    """Use 10 folder cross validation to select learning rate. """
    h_low = int(h_factor * s)
    h_high = int(h_factor * s * (1. + gamma))
    t_low = int(s)
    t_high = int(s * (1. + gamma))
    re_folding = _fold_split(sample_indices=range(len(y_tr)), k=num_fold)
    test_err_mat = np.zeros(shape=(len(lr_list), num_fold))
    # assume that x_star is unknown
    for lr_ind, lr in enumerate(lr_list):
        for ii, tr, te in re_folding:
            _, _, num_epochs_, run_time_, err_, x_hat = algo_graph_iht(
                x_mat=x_mat[tr], y_tr=y_tr[tr], max_epochs=max_epochs,
                lr=lr, x_star=None, x0=x0, tol_algo=tol_algo, edges=edges,
                costs=costs, g=g, root=root, h_low=h_low, h_high=h_high,
                t_low=t_low, t_high=t_high,
                proj_max_num_iter=proj_max_num_iter, verbose=verbose)
            err = np.linalg.norm(y_tr[te] - np.dot(x_mat[te], x_hat)) ** 2.
            test_err_mat[lr_ind][ii] += err
    min_index = np.argmin(np.mean(test_err_mat, axis=1))
    lr = lr_list[min_index]
    re = algo_graph_iht(
        x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs, lr=lr,
        x_star=x_star, x0=x0, tol_algo=tol_algo, edges=edges,
        costs=costs, g=g, root=root, h_low=h_low, h_high=h_high,
        t_low=t_low, t_high=t_high,
        proj_max_num_iter=proj_max_num_iter, verbose=verbose)
    return lr, re


def algo_graph_sto_iht(
        x_mat, y_tr, max_epochs, lr, x_star, x0, tol_algo, edges, costs, g,
        root, h_low, h_high, t_low, t_high, proj_max_num_iter, verbose, b,
        save_iter=False):
    """
    :param x_mat: the design matrix.
    :param y_tr: the array of measurements.
    :param max_epochs: the maximum epochs (iterations) allowed.
    :param lr: the learning rate should be 1.0 consistent with the paper.
    :param x_star: x_star is the true signal.
    :param x0: x0 is the initial point.
    :param tol_algo: tolerance parameter for early stopping.
    :param edges: edges in the graph.
    :param costs: edge costs
    :param g: number of connected component in the true signal.
    :param root: the root included in the result.
    :param h_low: lower bound of sparsity of head projection.
    :param h_high: upper bound of sparsity of head projection.
    :param t_low: lower bound of sparsity of tail projection.
    :param t_high: upper bound of sparsity of tail projection.
    :param proj_max_num_iter: maximum number of iterations of projection.
    :param verbose: print out some information.
    :param b: the block size
    :param save_iter: to save the iteration information
    :return:
    """
    # do not forget the random seed.
    np.random.seed()
    start_time = time.time()
    x_hat = x0
    (n, p) = x_mat.shape
    x_tr_t = np.transpose(x_mat)
    # if the block size is too large. just use single block
    b = n if n < b else b
    num_blocks = int(n) / int(b)
    prob = [1. / num_blocks] * num_blocks

    num_epochs = 0
    y_err_list = []
    x_err_list = []
    x_err_iter_list = []
    y_err_iter_list = []
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
            if save_iter:
                if x_star is not None:
                    x_err = np.linalg.norm(x_hat - x_star)
                    x_err_iter_list.append(x_err)
                y_err = np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) ** 2.
                y_err_iter_list.append(y_err)

        y_err_list.append(np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) ** 2.)
        if x_star is not None:
            x_err_list.append(np.linalg.norm(x_hat - x_star))

        if np.linalg.norm(x_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break

        # early stopping for the local optimal point.
        # algorithm will never improve the solution.
        if epoch_i >= 10 and \
                len(np.unique(y_err_list[-5:])) == 1 and \
                len(np.unique(x_err_list[-5:])) == 1:
            break

    run_time = time.time() - start_time
    if x_star is not None:
        return x_err_list, x_err_iter_list, y_err_list, y_err_iter_list, \
               num_epochs, run_time, x_err_list[-1], x_hat
    else:
        return x_err_list, x_err_iter_list, y_err_list, y_err_iter_list, \
               num_epochs, run_time, -1.0, x_hat


def cv_graph_sto_iht(x_mat, y_tr, b_list, lr_list, s, max_epochs, tol_algo,
                     x_star, x0, verbose, edges, costs, g, root,
                     proj_max_num_iter, h_factor, gamma, num_fold):
    """Use 10 folder cross validation to select learning rate. """
    h_low = int(h_factor * s)
    h_high = int(h_factor * s * (1. + gamma))
    t_low = int(s)
    t_high = int(s * (1. + gamma))
    re_folding = _fold_split(sample_indices=range(len(y_tr)), k=num_fold)
    test_err_mat = np.zeros(shape=(len(lr_list) * len(b_list), num_fold))
    para_dict, index = dict(), 0
    for lr, b in product(lr_list, b_list):
        for ii, tr, te in re_folding:
            _, _, _, _, _, run_time_, _, x_hat = algo_graph_sto_iht(
                x_mat=x_mat[tr], y_tr=y_tr[tr], max_epochs=max_epochs,
                lr=lr, x_star=None, x0=x0, tol_algo=tol_algo, edges=edges,
                costs=costs, g=g, root=root, h_low=h_low, h_high=h_high,
                t_low=t_low, t_high=t_high,
                proj_max_num_iter=proj_max_num_iter, verbose=verbose, b=b)
            err = np.linalg.norm(y_tr[te] - np.dot(x_mat[te], x_hat)) ** 2.
            test_err_mat[index][ii] += err
        para_dict[index] = (lr, b)
        index += 1
    min_index = np.argmin(np.mean(test_err_mat, axis=1))
    lr, b = para_dict[min_index]
    re = algo_graph_sto_iht(
        x_mat=x_mat, y_tr=y_tr, max_epochs=max_epochs, lr=lr, x_star=x_star,
        x0=x0, tol_algo=tol_algo, edges=edges, costs=costs, g=g, root=root,
        h_low=h_low, h_high=h_high, t_low=t_low, t_high=t_high,
        proj_max_num_iter=proj_max_num_iter, verbose=verbose, b=b)
    return lr, b, re


def algo_best_subset(
        x_mat, y_tr, max_epochs, lr, s, x_star, x0, tol_algo, verbose=0):
    """Best Subset a fake algorithm.
    :param x_mat: the design matrix.
    :param y_tr: the array of measurements.
    :param max_epochs: the maximum epochs (iterations) allowed.
    :param lr: the learning rate should be 1.0 consistent with the paper.
    :param s: the sparsity parameter.
    :param x_star: x_star is the true signal.
    :param x0: x0 is the initial point.
    :param tol_algo: tolerance parameter for early stopping.
    :param verbose: print out some info.
    :return: list of recovery error during training.
    """
    start_time = time.time()
    x_hat = x0
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y_tr)
    num_epochs = 0
    y_err_list = []
    x_err_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        # we obey the implementation used in their code
        bt = x_hat - lr * (np.dot(xtx, x_hat) - xty)
        x_hat = np.zeros_like(bt)
        x_hat[np.nonzero(x_star)[0]] = bt[np.nonzero(x_star)[0]]

        x_err_list.append(np.linalg.norm(x_hat - x_star))
        y_err_list.append(np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) ** 2.)

        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break

        # we reach a local minimum point.
        if len(np.unique(y_err_list[-5:])) == 1 and \
                len(np.unique(x_err_list[-5:])) == 1 and epoch_i >= 10:
            break
        if verbose > 0:
            print(epoch_i, s, x_err_list[-1])
    run_time = time.time() - start_time
    return x_err_list, y_err_list, num_epochs, run_time, x_err_list[-1]


def algo_graph_cosamp(
        x_mat, y_tr, max_epochs, x_star, x0, tol_algo, edges, costs,
        h_g, t_g, root, h_low, h_high, t_low, t_high,
        proj_max_num_iter, verbose):
    """
    :param x_mat:
    :param y_tr:
    :param max_epochs:
    :param x_star:
    :param x0:
    :param tol_algo:
    :param edges:
    :param costs:
    :param h_g:
    :param t_g:
    :param root:
    :param h_low:
    :param h_high:
    :param t_low:
    :param t_high:
    :param proj_max_num_iter:
    :param verbose:
    :return:
    """
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
    run_time = time.time() - start_time
    return x_err_list, y_err_list, num_epochs, run_time, x_err_list[-1]


def algo_cosamp(x_mat, y_tr, max_epochs, x_star, x0, tol_algo, s, verbose):
    """

    :param x_mat:
    :param y_tr:
    :param max_epochs:
    :param x_star:
    :param x0:
    :param tol_algo:
    :param s:
    :param verbose:
    :return:
    """
    start_time = time.time()
    x_hat = np.zeros_like(x0)
    x_tr_t = np.transpose(x_mat)
    m, p = x_mat.shape
    num_epochs = 0
    y_err_list = []
    x_err_list = []
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

        y_err_list.append(np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) ** 2.)
        x_err_list.append(np.linalg.norm(x_hat - x_star))

        if verbose > 0:
            print(epoch_i, s, x_err_list[-1])

        if np.linalg.norm(x_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, x_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return x_err_list, y_err_list, num_epochs, run_time, x_err_list[-1]


def algo_niht(x_mat, y_tr, max_epochs, s, x_star, x0, tol_algo, verbose):
    """Normalized Iterative Hard Thresholding proposed in [1].
    Reference:
        [1] Blumensath, Thomas, and Mike E. Davies. "Normalized iterative hard
            thresholding: Guaranteed stability and performance." IEEE Journal
            of selected topics in signal processing 4.2 (2010): 298-309.
    :param x_mat:
    :param y_tr:
    :param max_epochs:
    :param s:
    :param x_star:
    :param x0:
    :param tol_algo:
    :param verbose:
    :return: list of recovery error.
    """
    start_time = time.time()
    w_hat = x0
    c = 0.01
    kappa = 2. / (1 - c)
    (m, p) = x_mat.shape
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y_tr)
    gamma = np.argsort(np.abs(np.dot(x_tr_t, y_tr)))[-s:]

    num_epochs = 0
    y_err_list = []
    x_err_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        # we obey the implementation used in their code
        gn = xty - np.dot(xtx, w_hat)
        tmp_v = np.dot(x_mat[:, gamma], gn[gamma])
        xx = np.dot(gn[gamma], gn[gamma])
        yy = np.dot(tmp_v, tmp_v)
        if yy != 0:
            mu = xx / yy
        else:
            mu = 1.
        bt = w_hat + mu * gn
        bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
        w_tmp = bt
        gamma_next = np.nonzero(w_tmp)[0]
        if set(gamma_next).__eq__(set(gamma)):
            w_hat = w_tmp
        else:
            xx = np.linalg.norm(w_tmp - w_hat) ** 2.
            yy = np.linalg.norm(np.dot(x_mat, w_tmp - w_hat)) ** 2.
            if yy <= 0.0:
                continue
            if mu <= (1. - c) * xx / yy:
                w_hat = w_tmp
            elif mu > (1. - c) * xx / yy:
                while True:
                    mu = mu / (kappa * (1. - c))
                    bt = w_hat + mu * gn
                    bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
                    w_tmp = bt
                    xx = np.linalg.norm(w_tmp - w_hat) ** 2.
                    yy = np.linalg.norm(np.dot(x_mat, w_tmp - w_hat)) ** 2.
                    if yy <= 0.0:
                        break
                    if mu <= (1 - c) * xx / yy:
                        break
                gamma_next = np.nonzero(w_tmp)[0]
                w_hat = w_tmp
                gamma = gamma_next

        x_err_list.append(np.linalg.norm(w_hat - x_star))
        y_err_list.append(np.linalg.norm(y_tr - np.dot(x_mat, w_hat)) ** 2.)

        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_mat, w_hat)) <= tol_algo:
            break
        if verbose > 0:
            print(epoch_i, s, x_err_list[-1])
    run_time = time.time() - start_time
    return x_err_list, y_err_list, num_epochs, run_time, x_err_list[-1]
