# -*- coding: utf-8 -*-
__all__ = ['algo_test',
           'algo_iht',
           'algo_sto_iht',
           'algo_sto_iht_wrapper',
           'algo_da_sto_iht',
           'algo_head_sto_iht',
           'algo_best_subset',
           'algo_niht',
           'algo_graph_iht',
           'algo_graph_sto_iht',
           'algo_gradmp',
           'algo_sto_gradmp',
           'algo_graph_gradmp',
           'algo_graph_sto_gradmp',
           'algo_head_tail_binsearch',
           'wrapper_head_tail_binsearch']
import imp
import time
import numpy as np

try:
    import sparse_module

    try:
        from sparse_module import test
        from sparse_module import wrap_head_tail_binsearch
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')


def _expit(x):
    """ expit function. 1 /(1+exp(-x)) """
    if type(x) == np.float64:
        if x > 0.:
            return 1. / (1. + np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))
    out = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0.:
            out[i] = 1. / (1. + np.exp(-x[i]))
        else:
            out[i] = np.exp(x[i]) / (1. + np.exp(x[i]))
    return out


def _log_logistic(x):
    """ return log( 1/(1+exp(x)) )"""
    out = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            out[i] = -np.log(1 + np.exp(-x[i]))
        else:
            out[i] = x[i] - np.log(1 + np.exp(x[i]))
    return out


def _grad_w(x_tr, y_tr, wt, eta):
    """ return {+1,-1} Logistic (val,grad) on training samples. """
    assert len(wt) == (x_tr.shape[1] + 1)
    c, p = wt[-1], x_tr.shape[1]
    wt = wt[:p]
    yz = y_tr * (np.dot(x_tr, wt) + c)
    z = _expit(yz)
    loss = -np.sum(_log_logistic(yz)) + .5 * eta * np.dot(wt, wt)
    grad = np.zeros(p + 1)
    z0 = (z - 1) * y_tr
    grad[:p] = np.dot(x_tr.T, z0) + eta * wt
    grad[-1] = z0.sum()
    return loss, grad


def algo_test():
    x = np.arange(1, 13).reshape(3, 4)
    sum_x = test(np.asarray(x, dtype=np.double))
    print('sum: %.2f' % sum_x)


def algo_head_tail_binsearch(
        edges, w, costs, g, root, s_low, s_high, max_num_iter, verbose):
    prizes = w * w
    if s_high >= len(prizes) - 1:  # to avoid problem.
        s_high = len(prizes) - 1
    re_nodes = wrap_head_tail_binsearch(
        edges, prizes, costs, g, root, s_low, s_high, max_num_iter, verbose)
    proj_w = np.zeros_like(w)
    proj_w[re_nodes[0]] = w[re_nodes[0]]
    return re_nodes[0], proj_w


def algo_graph_sto_iht_linsearch(
        x_tr, y_tr, max_epochs, lr, w, w0, tol_algo, b, edges, costs, g,
        root, h_low, h_high, t_low, t_high, max_num_iter, verbose):
    start_time = time.time()
    np.random.seed()  # do not forget it.
    w_hat, num_iter = w0, 0
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    num_blocks = int(m) / int(b)
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for _ in range(max_epochs * num_blocks):
        num_iter = _
        # random select a block
        ii = np.random.randint(0, num_blocks)
        block = range(b * ii, b * (ii + 1))
        xtx = np.dot(x_tr_t[:, block], x_tr[block])
        xty = np.dot(x_tr_t[:, block], y_tr[block])
        gradient = -2. * (xty - np.dot(xtx, w_hat))
        fun_val_right = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        tmp_num_iter, adaptive_step, beta = 0, 2.0, 0.8
        while tmp_num_iter < 20:
            x_tmp = w_hat - adaptive_step * gradient
            fun_val_left = np.linalg.norm(y_tr - np.dot(x_tr, x_tmp)) ** 2.
            reg_term = adaptive_step / 2. * np.linalg.norm(gradient) ** 2.
            if fun_val_left > fun_val_right - reg_term:
                adaptive_step *= beta
            else:
                break
            tmp_num_iter += 1

        head_nodes, proj_grad = algo_head_tail_binsearch(
            edges, gradient, costs, g, root, h_low, h_high,
            max_num_iter, verbose)
        bt = w_hat - adaptive_step * proj_grad
        tail_nodes, proj_bt = algo_head_tail_binsearch(
            edges, bt, costs, g, root, t_low, t_high, max_num_iter, verbose)
        w_hat = proj_bt
        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)
        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_iter': num_iter,
            'run_time': run_time}


def algo_graph_da_iht(
        x_tr, y_tr, max_epochs, sigma, w, w0, tol_algo, b, edges, costs, g,
        root, h_low, h_high, t_low, t_high, max_num_iter, verbose):
    start_time = time.time()
    np.random.seed()  # do not forget it.
    w_hat, num_iter = w0, 0
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    num_blocks = int(m) / int(b)
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    grad, grad_bar = np.zeros_like(w), np.zeros_like(w)
    for _ in range(max_epochs * num_blocks):
        num_iter = _
        # random select a block
        ii = np.random.randint(0, num_blocks)
        block = range(b * ii, b * (ii + 1))
        xtx = np.dot(x_tr_t[:, block], x_tr[block])
        xty = np.dot(x_tr_t[:, block], y_tr[block])
        grad = -2. * (xty - np.dot(xtx, w_hat))
        grad_bar = (_ / (_ + 1.)) * grad_bar + (1. / (_ + 1.)) * grad
        grad_bar = (-1 / sigma) * grad_bar
        head_nodes, proj_grad = algo_head_tail_binsearch(
            edges, grad_bar, costs, g, root, h_low, h_high,
            max_num_iter, verbose)
        tail_nodes, proj_bt = algo_head_tail_binsearch(
            edges, proj_grad, costs, g, root, t_low,
            t_high, max_num_iter, verbose)
        w_hat = proj_bt

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)

        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_iter': num_iter,
            'run_time': run_time}


def algo_niht(x_tr, y_tr, max_epochs, s, w, w0, tol_algo):
    start_time = time.time()
    w_hat = w0
    c = 0.01
    kappa = 2. / (1 - c)
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    xtx, xty = np.dot(x_tr_t, x_tr), np.dot(x_tr_t, y_tr)
    gamma = np.argsort(np.abs(np.dot(x_tr_t, y_tr)))[-s:]
    num_epochs = 0
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        # we obey the implementation used in their code
        gn = xty - np.dot(xtx, w_hat)
        tmp_v = np.dot(x_tr[:, gamma], gn[gamma])
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
            yy = np.linalg.norm(np.dot(x_tr, w_tmp - w_hat)) ** 2.
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
                    yy = np.linalg.norm(np.dot(x_tr, w_tmp - w_hat)) ** 2.
                    if yy <= 0.0:
                        break
                    if mu <= (1 - c) * xx / yy:
                        break
                gamma_next = np.nonzero(w_tmp)[0]
                w_hat = w_tmp
                gamma = gamma_next

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)

        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'err': w_error_list[-1],
            'run_time': run_time}


def algo_graph_niht(
        x_tr, y_tr, max_epochs, w, w0, tol_algo, edges, costs, g, root,
        h_low, h_high, t_low, t_high, max_num_iter, verbose):
    start_time = time.time()
    w_hat = w0
    c = 0.01
    kappa = 2. / (1 - c)
    x_tr_t = np.transpose(x_tr)
    xtx, xty = np.dot(x_tr_t, x_tr), np.dot(x_tr_t, y_tr)
    gamma = np.argsort(np.abs(np.dot(x_tr_t, y_tr)))[-t_low:]
    num_epochs = 0
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        # we obey the implementation used in their code
        gn = xty - np.dot(xtx, w_hat)
        tmp_v = np.dot(x_tr[:, gamma], gn[gamma])
        mu = np.dot(gn[gamma], gn[gamma]) / np.dot(tmp_v, tmp_v)
        h_nodes, proj_grad = algo_head_tail_binsearch(
            edges, gn, costs, g, root, h_low, h_high, max_num_iter, verbose)
        bt = w_hat + mu * proj_grad
        t_nodes, proj_bt = algo_head_tail_binsearch(
            edges, bt, costs, g, root, t_low, t_high, max_num_iter, verbose)
        w_tmp = proj_bt
        gamma_next = np.nonzero(w_tmp)[0]
        if set(gamma_next).__eq__(set(gamma)):
            w_hat = w_tmp
        else:
            xx = np.linalg.norm(w_tmp - w_hat) ** 2.
            yy = np.linalg.norm(np.dot(x_tr, w_tmp - w_hat)) ** 2.
            if yy <= 0.0:
                break
            if mu <= (1. - c) * xx / yy:
                w_hat = w_tmp
            elif mu > (1. - c) * xx / yy:
                while True:
                    mu = mu / (kappa * (1. - c))
                    _, proj_grad = algo_head_tail_binsearch(
                        edges, gn, costs, g, root, h_low, h_high, max_num_iter,
                        verbose)
                    bt = w_hat + mu * proj_grad
                    _, proj_bt = algo_head_tail_binsearch(
                        edges, bt, costs, g, root, t_low, t_high, max_num_iter,
                        verbose)
                    w_tmp = proj_bt
                    xx = np.linalg.norm(w_tmp - w_hat) ** 2.
                    yy = np.linalg.norm(np.dot(x_tr, w_tmp - w_hat)) ** 2.
                    if yy <= 0.0:
                        break
                    if mu <= (1 - c) * xx / yy:
                        break
                gamma_next = np.nonzero(w_tmp)[0]
                w_hat = w_tmp
                gamma = gamma_next

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)

        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'err': w_error_list[-1],
            'run_time': run_time}


def algo_sto_iht(x_tr, y_tr, max_epochs, lr, s, x_star, x0, tol_algo, b,
                 with_replace=True, verbose=0):
    np.random.seed()  # do not forget the random seed.
    start_time = time.time()
    x_hat = x0
    (n, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    # if the block size is too large. just use a single block.
    b = n if n < b else b
    num_blocks = int(n) / int(b)
    prob = [1. / num_blocks] * num_blocks
    num_epochs = 0
    w_error_iter_list = []
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        rand_perm = np.random.permutation(num_blocks)
        for _ in range(num_blocks):
            ii = np.random.randint(0, num_blocks) if \
                with_replace else rand_perm[_]
            block = range(b * ii, b * (ii + 1))
            xtx = np.dot(x_tr_t[:, block], x_tr[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            gradient = - 2. * (xty - np.dot(xtx, x_hat))
            bt = x_hat - (lr / (prob[ii] * num_blocks)) * gradient
            bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
            x_hat = bt
            w_error = np.linalg.norm(x_hat - x_star)
            w_error_iter_list.append(w_error)
        w_error = np.linalg.norm(x_hat - x_star)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, x_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(x_star) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)
        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(x_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, x_hat)) <= tol_algo:
            break
        if verbose > 0:
            print('current epoch: %03d x_error: %.8e y_error: %.8e' %
                  (epoch_i, w_error, y_error))
        # we reach a local minimum point.
        if len(np.unique(y_error_list[-5:])) == 1 and \
                len(np.unique(w_error_list[-5:])) == 1 and epoch_i >= 10:
            break

    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'w_error_iter_list': w_error_iter_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'lr': lr,
            'err': w_error_iter_list[-1],
            'run_time': run_time}


def algo_iht(x_tr, y_tr, max_epochs, lr, s, x_star, x0, tol_algo):
    start_time = time.time()
    x_hat = x0
    (n, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    xtx, xty = np.dot(x_tr_t, x_tr), np.dot(x_tr_t, y_tr)
    num_epochs = 0
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        # we obey the implementation used in their code
        bt = x_hat - lr * (np.dot(xtx, x_hat) - xty)
        bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
        x_hat = bt
        w_error = np.linalg.norm(x_hat - x_star)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, x_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(x_star) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)

        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(x_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, x_hat)) <= tol_algo:
            break

        # we reach a local minimum point.
        if len(np.unique(y_error_list[-5:])) == 1 and \
                len(np.unique(w_error_list[-5:])) == 1 and epoch_i >= 10:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'lr': lr,
            'err': w_error_list[-1],
            'run_time': run_time}


def algo_best_subset(x_tr, y_tr, max_epochs, lr, w, w0, tol_algo):
    start_time = time.time()
    w_hat = w0
    x_tr_t = np.transpose(x_tr)
    xtx, xty = np.dot(x_tr_t, x_tr), np.dot(x_tr_t, y_tr)
    num_epochs = 0
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        # we obey the implementation used in their code
        bt = w_hat - lr * (np.dot(xtx, w_hat) - xty)
        w_hat = np.zeros_like(bt)
        w_hat[np.nonzero(w)[0]] = bt[np.nonzero(w)[0]]

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)

        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break

        # we reach a local minimum point.
        if len(np.unique(y_error_list[-5:])) == 1 and \
                len(np.unique(w_error_list[-5:])) == 1 and epoch_i >= 10:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'lr': lr,
            'err': w_error_list[-1],
            'run_time': run_time}


def algo_sto_iht_py(x_tr, y_tr, max_epochs, lr, s, w, w0, tol_algo, b,
                    with_replace=True, verbose=0):
    start_time = time.time()
    np.random.seed()  # do not forget it.
    w_hat = w0
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    # if the block size is too large. just use single block
    b = m if m < b else b
    num_blocks = int(m) / int(b)
    prob = [1. / num_blocks] * num_blocks
    num_epochs = 0
    w_error_iter_list = []
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        rand_perm = np.random.permutation(num_blocks)
        for ii in range(num_blocks):
            if not with_replace:
                ii = rand_perm[ii]
            else:
                ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            xtx = np.dot(x_tr_t[:, block], x_tr[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            gradient = - 2. * (xty - np.dot(xtx, w_hat))
            bt = w_hat - (lr / (prob[ii] * num_blocks)) * gradient
            bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
            w_hat = bt
            w_error = np.linalg.norm(w_hat - w)
            w_error_iter_list.append(w_error)
        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)
        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
        if verbose > 0:
            print('current epoch: %03d x_error: %.8e y_error: %.8e' %
                  (epoch_i, w_error, y_error))
        # we reach a local minimum point.
        if len(np.unique(y_error_list[-5:])) == 1 and \
                len(np.unique(w_error_list[-5:])) == 1 and epoch_i >= 10:
            break

    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'w_error_iter_list': w_error_iter_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'lr': lr,
            'err': w_error_iter_list[-1],
            'run_time': run_time}


def algo_head_sto_iht(x_tr, y_tr, max_epochs, lr, s, w, w0, tol_algo, b,
                      head_s, with_replace=True, verbose=0):
    start_time = time.time()
    np.random.seed()  # do not forget it.
    w_hat = w0
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    # if the block size is too large. just use single block
    b = m if m < b else b
    num_blocks = int(m) / int(b)
    prob = [1. / num_blocks] * num_blocks
    num_epochs = 0
    w_error_iter_list = []
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        rand_perm = np.random.permutation(num_blocks)
        for ii in range(num_blocks):
            if not with_replace:
                ii = rand_perm[ii]
            else:
                ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            xtx = np.dot(x_tr_t[:, block], x_tr[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            gradient = - 2. * (xty - np.dot(xtx, w_hat))
            gradient[np.argsort(np.abs(gradient))[0:p - head_s]] = 0.
            bt = w_hat - (lr / (prob[ii] * num_blocks)) * gradient
            bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
            w_hat = bt
            w_error = np.linalg.norm(w_hat - w)
            w_error_iter_list.append(w_error)
        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)
        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
        if verbose > 0:
            print('current epoch: %03d x_error: %.8e y_error: %.8e' %
                  (epoch_i, w_error, y_error))
        # we reach a local minimum point.
        if len(np.unique(y_error_list[-5:])) == 1 and \
                len(np.unique(w_error_list[-5:])) == 1 and epoch_i >= 10:
            break

    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'w_error_iter_list': w_error_iter_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'lr': lr,
            'err': w_error_iter_list[-1],
            'run_time': run_time}


def algo_da_sto_iht(x_tr, y_tr, max_epochs, lr, s, w, w0, tol_algo, b,
                    head_s, with_replace=True, verbose=0):
    start_time = time.time()
    np.random.seed()  # do not forget it.
    w_hat = np.copy(w0)
    grad_bar = np.copy(w0)
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    # if the block size is too large. just use single block
    b = m if m < b else b
    num_blocks = int(m) / int(b)
    prob = [1. / num_blocks] * num_blocks
    num_epochs = 0
    w_error_iter_list = []
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    tt = 0.0
    for epoch_i in range(max_epochs):
        num_epochs += 1
        rand_perm = np.random.permutation(num_blocks)
        for ii in range(num_blocks):
            if not with_replace:
                ii = rand_perm[ii]
            else:
                ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            xtx = np.dot(x_tr_t[:, block], x_tr[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            gradient = - 2. * (xty - np.dot(xtx, w_hat))
            grad_bar = tt / (tt + 1.) * grad_bar + 1. / (tt + 1.) * gradient
            gradient[np.argsort(np.abs(grad_bar))[0:p - head_s]] = 0.
            bt = w_hat - (lr / (prob[ii] * num_blocks)) * gradient
            bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
            w_hat = bt
            w_error = np.linalg.norm(w_hat - w)
            w_error_iter_list.append(w_error)
        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)
        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
        if verbose > 0:
            print('current epoch: %03d x_error: %.8e y_error: %.8e' %
                  (epoch_i, w_error, y_error))
        # we reach a local minimum point.
        if len(np.unique(y_error_list[-5:])) == 1 and \
                len(np.unique(w_error_list[-5:])) == 1 and epoch_i >= 10:
            break

    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'w_error_iter_list': w_error_iter_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'lr': lr,
            'err': w_error_iter_list[-1],
            'run_time': run_time}


def algo_sto_iht_wrapper(
        x_tr, y_tr, max_epochs, lr, s, x_star, x0, tol_algo, b,
        with_replace=True, loss_func=0, verbose=0, lambda_l2=0.0,
        tol_rec=1e-6, use_py=False):
    if use_py:
        return algo_sto_iht_py(
            x_tr, y_tr, max_epochs, lr, s, x_star, x0, tol_algo, b,
            with_replace, verbose)
    else:
        try:
            from sparse_module import algo_sto_iht
            n, p = np.shape(x_tr)
            num_blocks = int(n / b)
            prob_arr = np.asarray([1.] * num_blocks) / float(num_blocks)
            re = algo_sto_iht(
                x_tr, y_tr, x0, x_star, prob_arr, n, p, s, b, loss_func,
                max_epochs, with_replace, verbose, lr, lambda_l2, tol_algo,
                tol_rec)
            return {'x_hat': re[0],
                    'x_bar': re[1],
                    'epochs_losses': re[2],
                    'x_epochs_errors': re[3],
                    'run_time': re[4],
                    'num_epochs': re[5],
                    'err': re[3][re[5] - 1]}
        except ImportError:
            print('error')


def algo_graph_iht(
        x_tr, y_tr, max_epochs, lr, x_star, x0, tol_algo, edges, costs, g,
        root, h_low, h_high, t_low, t_high, proj_max_num_iter, verbose):
    start_time = time.time()
    w_hat, num_iter = x0, 0
    x_tr_t = np.transpose(x_tr)
    xtx, xty = np.dot(x_tr_t, x_tr), np.dot(x_tr_t, y_tr)
    num_epochs = 0
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        grad = -1. * (xty - np.dot(xtx, w_hat))
        head_nodes, proj_gradient = algo_head_tail_binsearch(
            edges, grad, costs, g, root, h_low, h_high,
            proj_max_num_iter, verbose)
        bt = w_hat - lr * proj_gradient
        tail_nodes, proj_bt = algo_head_tail_binsearch(
            edges, bt, costs, g, root, t_low, t_high,
            proj_max_num_iter, verbose)
        w_hat = proj_bt

        w_error = np.linalg.norm(w_hat - x_star)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(x_star) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)
        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
        # we reach a local minimum point.
        if len(np.unique(y_error_list[-5:])) == 1 and \
                len(np.unique(w_error_list[-5:])) == 1 and epoch_i >= 10:
            break

    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'lr': lr,
            'err': w_error_list[-1],
            'run_time': run_time}


def algo_graph_sto_iht(
        x_tr, y_tr, max_epochs, lr, x_star, x0, tol_algo, b, edges, costs, g,
        root, h_low, h_high, t_low, t_high, proj_max_num_iter,
        verbose, with_replace=True):
    start_time = time.time()
    np.random.seed()  # do not forget it.
    w_hat = x0
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    # if the block size is too large. just use single block
    b = m if m < b else b
    num_blocks = int(m) / int(b)
    prob = [1. / num_blocks] * num_blocks
    y_error_list = []
    w_error_list = []
    w_error_iter_list = []
    w_error_ratio_list = []
    num_epochs = 0
    for epoch_i in range(max_epochs):
        rand_perm = np.random.permutation(num_blocks)
        for _ in range(num_blocks):
            ii = np.random.randint(0, num_blocks) if \
                with_replace else rand_perm[_]
            block = range(b * ii, b * (ii + 1))
            xtx = np.dot(x_tr_t[:, block], x_tr[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            gradient = -2. * (xty - np.dot(xtx, w_hat))
            head_nodes, proj_grad = algo_head_tail_binsearch(
                edges, gradient, costs, g, root, h_low, h_high,
                proj_max_num_iter, verbose)
            bt = w_hat - (lr / (prob[ii] * num_blocks)) * proj_grad
            tail_nodes, proj_bt = algo_head_tail_binsearch(
                edges, bt, costs, g, root,
                t_low, t_high, proj_max_num_iter, verbose)
            w_hat = proj_bt
            w_error = np.linalg.norm(w_hat - x_star)
            w_error_iter_list.append(w_error)

        num_epochs += 1
        w_error = np.linalg.norm(w_hat - x_star)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(x_star) ** 2.)
        w_error_list.append(w_error)
        y_error_list.append(y_error)
        w_error_ratio_list.append(w_error_ratio)

        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break

        # we reach a local minimum point.
        if len(np.unique(y_error_list[-5:])) == 1 and \
                len(np.unique(w_error_list[-5:])) == 1 and epoch_i >= 10:
            break

    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'w_error_iter_list': w_error_iter_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'lr': lr,
            'err': w_error_iter_list[-1],
            'run_time': run_time}


def algo_graph_svrg(
        x_tr, y_tr, max_epochs, lr, w, w0, tol_algo, b, edges, costs, g,
        root, h_low, h_high, t_low, t_high, max_num_iter, verbose):
    start_time = time.time()
    np.random.seed()  # do not forget it.
    w_hat = w0
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    # if the block size is too large. just use single block
    b = m if m < b else b
    num_blocks = int(m) / int(b)
    y_error_list = []
    w_error_list = []
    w_error_iter_list = []
    w_error_ratio_list = []
    num_epochs = 0
    w_hat_pre = np.zeros_like(w)
    xtx = np.dot(x_tr_t, x_tr)
    xty = np.dot(x_tr_t, y_tr)
    for s in range(max_epochs):
        w_hat = np.copy(w_hat_pre)
        mu_hat = 1. / num_blocks * (-2. * (xty - np.dot(xtx, w_hat)))
        wt = np.copy(w_hat)
        for num_iter_i in range(5 * num_blocks):
            # random select a block
            ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            xtx = np.dot(x_tr_t[:, block], x_tr[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            gradient_01 = -2. * (xty - np.dot(xtx, wt))
            gradient = -2. * (xty - np.dot(xtx, wt))
            head_nodes, proj_grad = algo_head_tail_binsearch(
                edges, gradient, costs, g, root, h_low, h_high,
                max_num_iter, verbose)
            bt = w_hat - lr * proj_grad
            tail_nodes, proj_bt = algo_head_tail_binsearch(
                edges, bt, costs, g, root,
                t_low, t_high, max_num_iter, verbose)
            w_hat = proj_bt
            w_error = np.linalg.norm(w_hat - w)
            w_error_iter_list.append(w_error)

        num_epochs += 1
        w_error = np.linalg.norm(w_hat - w)
        print('num_epoch: %02d m: %04d b: %02d error(w):%.4f ' %
              (num_epochs, m, b, w_error))
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        w_error_list.append(w_error)
        y_error_list.append(y_error)
        w_error_ratio_list.append(w_error_ratio)

        # print('m: %03d w_error: %.6e' % (len(x_tr), w_error))
        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'w_error_iter_list': w_error_iter_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'lr': lr,
            'err': w_error_iter_list[-1],
            'run_time': run_time}


def algo_gradmp(x_tr, y_tr, max_epochs, w, w0, tol_algo, s):
    start_time = time.time()
    w_hat = np.zeros_like(w0)
    x_tr_t = np.transpose(x_tr)
    m, p = x_tr.shape
    num_epochs = 0
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    xtx, xty = np.dot(x_tr_t, x_tr), np.dot(x_tr_t, y_tr)
    for epoch_i in range(max_epochs):
        num_epochs += 1
        grad = -(2. / float(m)) * (np.dot(xtx, w_hat) - xty)  # proxy
        gamma = np.argsort(abs(grad))[-2 * s:]  # identify
        gamma = np.union1d(w_hat.nonzero()[0], gamma)
        bt = np.zeros_like(w_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_tr[:, gamma]), y_tr)
        gamma = np.argsort(abs(bt))[-s:]
        w_hat = np.zeros_like(w_hat)
        w_hat[gamma] = bt[gamma]

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)

        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'err': w_error_list[-1],
            'run_time': run_time}


def algo_sto_gradmp(x_tr, y_tr, max_epochs, w, w0, tol_algo, s, b):
    start_time = time.time()
    w_hat = np.zeros_like(w0)
    x_tr_t = np.transpose(x_tr)
    m, p = x_tr.shape
    # if the block size is too large. just use single block
    b = m if m < b else b
    num_blocks = int(m) / int(b)
    num_epochs = 0
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        for num_iter_i in range(num_blocks):
            # random select a block
            ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))

            xtx = np.dot(x_tr_t[:, block], x_tr[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            # proxy
            grad = -2. * (np.dot(xtx, w_hat) - xty)
            # identify
            gamma = np.argsort(abs(grad))[-2 * s:]
            gamma = np.union1d(w_hat.nonzero()[0], gamma)
            bt = np.zeros_like(w_hat)
            bt[gamma] = np.dot(np.linalg.pinv(x_tr[:, gamma]), y_tr)
            gamma = np.argsort(abs(bt))[-s:]
            w_hat = np.zeros_like(w_hat)
            w_hat[gamma] = bt[gamma]
        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)

        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'run_time': run_time}


def algo_graph_gradmp(
        x_tr, y_tr, max_epochs, w, w0, tol_algo, edges, costs, h_g, t_g, root,
        h_low, h_high, t_low, t_high, max_num_iter, verbose):
    start_time = time.time()
    w_hat = np.zeros_like(w0)
    x_tr_t = np.transpose(x_tr)
    xtx, xty = np.dot(x_tr_t, x_tr), np.dot(x_tr_t, y_tr)
    num_epochs = 0
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        grad = -2. * (np.dot(xtx, w_hat) - xty)  # proxy
        head_nodes, proj_grad = algo_head_tail_binsearch(
            edges, grad, costs, h_g, root,
            h_low, h_high, max_num_iter, verbose)
        gamma = np.union1d(w_hat.nonzero()[0], head_nodes)
        bt = np.zeros_like(w_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_tr[:, gamma]), y_tr)
        tail_nodes, proj_bt = algo_head_tail_binsearch(
            edges, bt, costs, t_g, root,
            t_low, t_high, max_num_iter, verbose)
        w_hat = proj_bt

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'err': w_error_list[-1],
            'run_time': run_time}


def algo_graph_sto_iht_2(para):
    x_tr, y_tr, max_epochs, lr, s, w, w0, tol_algo, tol_rec, b, \
    edges, costs, g, root, head_sparsity_low, head_sparsity_high, \
    tail_sparsity_low, tail_sparsity_high, max_num_iter, verbose, \
    data_index = para
    if tail_sparsity_high <= s:
        print('tail_sparsity_high is at least higher than %d' % s)
        exit()
    np.random.seed()  # do not forget it.
    w_hat = w0
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    num_blocks = int(m) / int(b)
    max_iter = max_epochs * num_blocks
    for _ in range(max_iter):
        ii = np.random.randint(0, num_blocks)  # random select a block
        block = range(b * ii, b * (ii + 1))
        xtx = np.dot(x_tr_t[:, block], x_tr[block])
        xty = np.dot(x_tr_t[:, block], y_tr[block])
        gradient = -1. * (xty - np.dot(xtx, w_hat))
        prizes = gradient * gradient
        head_nodes = algo_head_tail_binsearch(
            edges, prizes, costs, g, root,
            head_sparsity_low, head_sparsity_high, max_num_iter, verbose)
        grad_head = np.zeros_like(gradient)
        grad_head[head_nodes] = gradient[head_nodes]
        bt = w_hat - lr * grad_head
        prizes = bt * bt
        tail_nodes = algo_head_tail_binsearch(
            edges, prizes, costs, g, root,
            tail_sparsity_low, tail_sparsity_high,
            max_num_iter, verbose)
        wt_tail = np.zeros_like(bt)
        wt_tail[tail_nodes] = bt[tail_nodes]
        w_hat = wt_tail
        error = (np.linalg.norm(w_hat - w) ** 2.) / (np.linalg.norm(w) ** 2.)
        print(np.linalg.norm(y_tr - np.dot(x_tr, w_hat)), error)
        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    error = (np.linalg.norm(w_hat - w) ** 2.) / (np.linalg.norm(w) ** 2.)
    if error <= 5e-2:
        return data_index, 1.0
    else:
        if m >= 150:
            print('error: %.6f m: %d' % (np.linalg.norm(w_hat - w), m))
        return data_index, 0.0


def algo_graph_sto_iht_backtracking(para):
    x_tr, y_tr, max_epochs, lr, s, w, w0, tol_algo, tol_rec, b, \
    edges, costs, g, root, head_sparsity_low, head_sparsity_high, \
    tail_sparsity_low, tail_sparsity_high, max_num_iter, verbose, \
    data_index = para
    if tail_sparsity_high <= s:
        print('tail_sparsity_high is at least higher than %d' % s)
        exit()
    np.random.seed()  # do not forget it.
    w_hat = w0
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    num_blocks = int(m) / int(b)
    max_iter = max_epochs * num_blocks
    for _ in range(max_iter):
        ii = np.random.randint(0, num_blocks)  # random select a block
        block = range(b * ii, b * (ii + 1))
        xtx = np.dot(x_tr_t[:, block], x_tr[block])
        xty = np.dot(x_tr_t[:, block], y_tr[block])
        gradient = -2. * (xty - np.dot(xtx, w_hat))
        fun_val_right = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        tmp_num_iter, adaptive_step, beta = 0, 1.0, 0.8
        while tmp_num_iter < 20:
            x_tmp = w_hat - adaptive_step * gradient
            fun_val_left = np.linalg.norm(y_tr - np.dot(x_tr, x_tmp)) ** 2.
            reg_term = adaptive_step / 2. * np.linalg.norm(gradient) ** 2.
            if fun_val_left > fun_val_right - reg_term:
                adaptive_step *= beta
            else:
                break
            tmp_num_iter += 1

        prizes = gradient * gradient
        head_nodes = algo_head_tail_binsearch(
            edges, prizes, costs, g, root,
            head_sparsity_low, head_sparsity_high, max_num_iter, verbose)
        grad_head = np.zeros_like(gradient)
        grad_head[head_nodes] = gradient[head_nodes]
        bt = w_hat - adaptive_step * grad_head
        prizes = bt * bt
        tail_nodes = algo_head_tail_binsearch(
            edges, prizes, costs, g, root,
            tail_sparsity_low, tail_sparsity_high,
            max_num_iter, verbose)
        wt_tail = np.zeros_like(bt)
        wt_tail[tail_nodes] = bt[tail_nodes]
        w_hat = wt_tail

        error = (np.linalg.norm(w_hat - w) ** 2.) / (np.linalg.norm(w) ** 2.)
        print(np.linalg.norm(y_tr - np.dot(x_tr, w_hat)), error)
        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    error = (np.linalg.norm(w_hat - w) ** 2.) / (np.linalg.norm(w) ** 2.)
    if error <= 5e-2:
        return data_index, 1.0
    else:
        if m >= 150:
            print('error: %.6f m: %d' % (np.linalg.norm(w_hat - w), m))
        return data_index, 0.0


def algo_graph_sto_gradmp(
        x_tr, y_tr, max_epochs, w, w0, tol_algo, b, edges, costs, g, root,
        h_low, h_high, t_low, t_high, max_num_iter, verbose):
    start_time = time.time()
    w_hat = np.zeros_like(w0)
    x_tr_t = np.transpose(x_tr)
    m, p = x_tr.shape
    # if the block size is too large. just use single block
    b = m if m < b else b
    num_blocks = int(m) / int(b)

    num_epochs = 0
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        for num_iter_i in range(num_blocks):
            # random select a block
            ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            xtx = np.dot(x_tr_t[:, block], x_tr[block])
            xty = np.dot(x_tr_t[:, block], y_tr[block])
            grad = -2. * (np.dot(xtx, w_hat) - xty)  # proxy
            head_nodes, proj_grad = algo_head_tail_binsearch(  # identify
                edges, grad, costs, g, root,
                h_low, h_high, max_num_iter, verbose)
            gamma = np.union1d(w_hat.nonzero()[0], head_nodes)
            bt = np.zeros_like(w_hat)
            bt[gamma] = np.dot(np.linalg.pinv(x_tr[:, gamma]), y_tr)
            tail_nodes, proj_bt = algo_head_tail_binsearch(
                edges, bt, costs, g, root, t_low, t_high,
                max_num_iter, verbose)
            w_hat = proj_bt

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_epochs': num_epochs,
            'run_time': run_time}


def wrapper_head_tail_binsearch(
        edges, prizes, costs, g, root, sparsity_low, sparsity_high,
        max_num_iter, verbose):
    return wrap_head_tail_binsearch(
        edges, prizes, costs, g, root, sparsity_low, sparsity_high,
        max_num_iter, verbose)
