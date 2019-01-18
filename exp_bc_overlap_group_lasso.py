# -*- coding: utf-8 -*-
import os
import csv
import time
import pickle
from os import sys, path
import numpy as np
from itertools import product
from numpy.random import normal
import multiprocessing


def is_member(a, b):
    # None can be replaced by any other "not in b" value
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]


def expand_data(data):
    x = data['x']
    groups = data['groups']
    p = np.shape(x)[1]
    # First get the size of the new matrix
    ep = 0
    for g in range(len(groups)):
        ep = ep + len(groups[g])
    e_groups = {i: dict() for i in range(len(groups))}
    exp_2_orig = np.zeros(p, ep)
    c = 0
    start = 0
    for g in range(len(groups)):
        e_groups[g] = range(start, start + len(groups[g]) - 1)
        start = start + len(groups[g])
        for i in groups[g]:
            c += 1
            exp_2_orig[:, c] = [np.sum(_ == i) for _ in range(p)]
    ex = x * exp_2_orig
    e_data = data
    e_data['x'] = ex
    e_data['groups'] = e_groups
    e_data['exp_2_orig'] = exp_2_orig
    return e_data


def blr_oracle(b, w, s, data, at_zero, cp, cn):
    """
    Compute the balanced logistic regression function and its gradient
    :param b:       is the offset, w the linear function.
    :param w:       weight of variables.
    :param s:       is the index of a group (0 for the offset)
    :param data:    contains X, Y and groups.
    :param at_zero: if 1, the loss will be computed for the same w, but with
                    the group s set to 0.
    :param cp:
    :param cn:
    :return: f is the value of the function.
            grad is a column vector containing the partial gradient with
            respect to group s.
    """
    x = data['x']
    y = data['y']
    groups = data['groups']
    n = len(y)
    posi_idx = np.where(y == 1)
    nega_idx = np.where(y == -1)
    if np.empty(w):
        exp_term = np.exp(-np.dot(y, b))
    else:
        w_loc = w
        if at_zero:  # Evaluate loss/grad for w(group)=0
            w_loc[groups[s]] = 0
        exp_term = np.exp(-y * (np.dot(x, w_loc) + b))
    term1 = cp * np.sum(np.log(1. + exp_term(posi_idx)))
    term2 = cn * np.sum(np.log(1 + exp_term(nega_idx)))
    f_val = (term1 + term2) / n
    y[posi_idx] = cp * y[posi_idx]
    y[nega_idx] = cn * y[nega_idx]
    if s:
        x_trans = np.transpose(x[:, groups[s]])
        f_grad = np.dot(x_trans, -y * exp_term / (1 + exp_term)) / n
    else:
        f_grad = np.sum(-y * exp_term / (1. + exp_term)) / n
    return f_val, f_grad


def blr_h(b, w, s, data, cp, cn):
    x = data['x']
    y = data['y']
    groups = data['groups']
    n = len(y)
    posi_idx = np.where(y == 1)
    nega_idx = np.where(y == -1)
    xp = x[posi_idx, :]
    xn = x[nega_idx, :]
    yp = y[posi_idx]
    yn = y[nega_idx]
    if np.empty(w):
        p_exp_term = np.exp(-yp * b)
        n_exp_term = np.exp(-yn * b)
    else:
        p_exp_term = np.exp(-yp * (np.dot(xp, w) + b))
        n_exp_term = np.exp(-yn * (np.dot(xn, w) + b))
    if s:  # diagonal of block hessian
        xp_trans = np.transpose(cp * (xp[:, groups[s]] ** 2.0))
        term1 = np.dot(xp_trans, p_exp_term / ((1.0 + p_exp_term) ** 2.0))
        xn_trans = np.transpose(cn * (xn[:, groups[s]] ** 2.0))
        term2 = np.dot(xn_trans, n_exp_term / ((1.0 + n_exp_term) ** 2.0))
        diag_h = (term1 + term2) / n
        h = np.diag(diag_h)
    else:
        term1 = cp * np.sum(p_exp_term / ((1.0 + p_exp_term) ** 2.0))
        term2 = cn * np.sum(n_exp_term / ((1.0 + n_exp_term) ** 2.0))
        h = (term1 + term2) / n
    return h


def blr_gl(b, w, data, cp, cn):
    x = data['x']
    y = data['y']
    n = len(y)
    posi_idx = np.where(y == 1)
    nega_idx = np.where(y == -1)
    if np.empty(w):
        exp_term = np.exp(-y * b)
    else:
        exp_term = np.exp(-y * (np.dot(x, w) + b))
    y[posi_idx] = cp * y[posi_idx]
    y[nega_idx] = cn * y[nega_idx]
    f_grad = np.dot(np.transpose(x), -y * exp_term / (1.0 + exp_term)) / n
    return f_grad


def cv_split_bal(y, n_folds):
    splits = dict()
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]
    np_, nn_ = len(pos_idx), len(neg_idx)
    p_size_fold = np.round(len(pos_idx) / n_folds)
    n_size_fold = np.round(len(neg_idx) / n_folds)
    pa = pos_idx[np.random.permutation(len(pos_idx))]
    na = neg_idx[np.random.permutation(len(neg_idx))]
    for i in range(1, n_folds + 1):
        splits[i] = dict()
        if i < n_folds:
            posi = pa[0:(i - 1) * p_size_fold]
            posi = np.append(posi, pa[i * p_size_fold:np_])
            nega = na[0:(i - 1) * n_size_fold]
            nega = np.append(nega, na[(i * n_size_fold):nn_])
            splits[i]['x_tr'] = np.append(posi, nega)
            # leave one folder
            posi = pa[((i - 1) * p_size_fold):(i * p_size_fold)]
            nega = na[((i - 1) * n_size_fold):(i * n_size_fold)]
            splits[i]['x_te'] = np.append(posi, nega)
        else:
            posi = pa[1:(i - 1) * p_size_fold]
            nega = na[1:(i - 1) * n_size_fold]
            splits[i]['x_tr'] = np.append(posi, nega)
            posi = pa[(i - 1) * p_size_fold:np_]
            nega = na[(i - 1) * n_size_fold:nn_]
            splits[i]['x_te'] = np.append(posi, nega)
    return splits


def get_data(type_, nfolds, dparam):
    data, splits = None, None
    if type_ == 'chain':
        dparam = 0
        pass
    elif type_ == 'vv':
        raw_data = pickle.load(open(root_input + 'data.pkl'))
        x = raw_data['x']
        y = raw_data['y']
        entrez = raw_data['entrez']
        sub_splits = dict()
        super_splits = cv_split_bal(y, nfolds[0])
        for i in range(1, nfolds[0] + 1):
            sub_splits[i] = cv_split_bal(y[super_splits[i]['x_tr']], nfolds[1])
        data = dict()
        data['x'] = x
        data['y'] = y
        data['entrez'] = entrez
        splits = dict()
        splits['splits'] = super_splits
        splits['sub_splits'] = sub_splits
    return data, splits


def process(data, type_):
    p_data = dict()
    if type_ == 'correl':
        p = data['x'].shape[1]
        r = np.zeros(p)
        for i in range(p):
            c = np.corrcoef(data['x'][:, i], data['y'])
            r[i] = c[0][1]
        k_idx = np.argsort(np.abs(r))[-500:]
        p_data['y'] = data['y']
        p_data['x'] = data['x'][:, k_idx]
        p_data['entrez'] = data['entrez'][k_idx]
    elif type_ == 'threshold':
        k_idx = []
    else:
        print('error: unknown pre-processing.')
        k_idx = []
    groups = data['groups']
    c = 0
    p_groups = dict()
    k_groups = dict()
    k_groups_idx = dict()
    for g in range(len(groups)):
        rv = groups[g][[np.sum(_ == k_idx) for _ in groups[g]]]
        if not np.empty(rv):
            c = c + 1
            tmp = np.zeros(len(rv))
            i = 0
            for v in rv:
                i = i + 1
                tmp[i] = np.where(k_idx == v)
            p_groups[c] = tmp
            k_groups[c] = rv
            k_groups_idx = g
    p_data['groups'] = p_groups
    print('[process] Pre-process over. Kept %d(/%d) variables '
          'and %d(/%d) groups', len(k_idx), np.shape(data['x'])[1],
          len(p_groups), len(groups))
    return p_data, k_idx, k_groups, k_groups_idx


def tseng_newton(xc, f, b, w, ds, jh, jg, j_fc, s, lambda_, data):
    """
        This function just implements the optimization of Tseng et al.
        on a given group.
        Reference:
        [1]     Tseng, Paul, and Sangwoon Yun. "A coordinate gradient descent
                method for nonsmooth separable minimization."
                Mathematical Programming 117.1-2 (2009): 387-423.
    :param xc:
    :param f:
    :param b:
    :param w:
    :param ds:
    :param jh:
    :param jg:
    :param j_fc:
    :param s:
    :param lambda_:
    :param data:
    :return:
    """
    thresh = 1e-1
    coeff = 0.5
    max_itr = 30
    groups = data['groups']
    w_eval = w
    #  Case of a feature (not the offset) Compute the direction
    if s:
        if np.linalg.norm(xc) == 0 and np.linalg.norm(jg) < ds * lambda_:
            xt = xc
            return xt
        h_max = np.max([np.max(np.diag(jh)), 1e-16])
        u = jg - h_max * xc
        if np.linalg.norm(u) < ds * lambda_:
            d = -xc
        else:
            u = u / np.linalg.norm(u)
            d = -(jg - lambda_ * ds * u) / h_max
        alpha = 1000
        jg_d = np.dot(jg, d)
        norm_xc = np.linalg.norm(xc)
        if norm_xc > 0:
            gcd = jg_d + lambda_ * ds * (np.dot(np.transpose(xc) / norm_xc, d))
        else:
            gcd = jg_d
        if gcd > 0:
            print('d is not a descent direction')
        # Line search with Armijo condition
        nxc = np.linalg.norm(xc)
        delta = jg_d + lambda_ * ds * (np.linalg.norm(xc + d) - nxc)
        itr = 1
        xt = xc + alpha * d
        w_eval[groups[s]] = xt
        j_ft = f(b, w_eval, s, data, 0)
        norm_xt = np.linalg.norm(xt)
        wolfe = np.where(j_ft - j_fc + lambda_ * ds * (
                norm_xt - nxc) <= alpha * thresh * delta)
        while (not wolfe) and itr < max_itr:
            alpha = alpha * coeff
            xt = xc + alpha * d
            w_eval[groups[s]] = xt
            j_ft = f(b, w_eval, s, data, 0)
            norm_xt = np.linalg.norm(xt)
            wolfe = np.where(j_ft - j_fc + lambda_ * ds * (
                    norm_xt - nxc) <= alpha * thresh * delta)
            itr = itr + 1
    else:  # Case of the offset Compute the direction
        d = -np.linalg.solve(jh, jg)
        alpha = 1
        j_gd = np.dot(np.transpose(jg), d)
        if j_gd > 0:
            print('d is not a descent direction')
        # Line search with Armijo condition
        delta = j_gd
        itr = 1
        xt = xc + alpha * d
        j_ft = f(xt, w, s, data, 0)
        wolfe = np.where(j_ft - j_fc <= (alpha * thresh * delta))
        while (not wolfe) and (itr < max_itr):
            alpha = alpha * coeff
            xt = xc + alpha * d
            j_ft = f(xt, w, s, data, 0)
            wolfe = np.where(j_ft - j_fc <= (alpha * thresh * delta))
            itr = itr + 1
    if itr == max_itr:
        print('maximal number of iterations reached in armijo')
    return xt


def block_tseng_as(f, f_h, f_gl, b_w0, as_, data, lambda_, d, options):
    """
    Active set version of Tseng et al. algorithm. Just calls block_tseng on a
    restricted set of variables, then check if the solution is a global
    optimum, and if not, adds the groups whose gradient norm is larger than
    lambda.
    :param f:
    :param f_h:
    :param f_gl:
    :param b_w0:
    :param as_:
    :param data:
    :param lambda_:
    :param d:
    :param options:
    :return:
    """
    n_add = 100
    b = b_w0[0]  # intercept
    ww = b_w0[1:]
    groups = data['groups']
    grid_x = range(len(groups))
    ngc = np.zeros(len(grid_x))
    # s_idx must contain the indices of the variables from all the groups
    # in the active set.
    s_idx = [data['groups'][g] for g in as_]
    global_sol = 0
    first = 1
    run = 0
    verbose = 0
    next_grad = None
    # Iterate while no global solution is found and the number of selected
    # variables is less than maxFeat
    while (not global_sol) and (len(s_idx) <= options['max_feat']):
        # Build a data set restricted to the variables in AS
        if verbose > 0:
            print('[block_tseng_as] Running optimization with '
                  '%d groups in the active set', len(as_))
        sub_data = dict()
        sub_data['x'] = data['x'][:, s_idx]
        sub_data['y'] = data['y']
        sub_data['groups'] = {i: dict() for i in range(len(as_))}
        c = 0
        for g in as_:
            c = c + 1
            members = [np.sum(_ == data['groups'][g]) for _ in s_idx]
            sub_data['groups'][c] = np.where(members)

        # Optimize group lasso on the restricted dataset.
        b, w = block_tseng(f, f_h, b, ww[s_idx], sub_data, lambda_, d[as_],
                           options)
        ww = np.zeros(np.shape(data['x'])[0])
        ww[s_idx] = w
        ngc = np.zeros(len(grid_x))
        # Recompute the loss gradient at the new W.
        if verbose > 0:
            print('[block_tseng_as] Recomputing the full gradient.')
        full_gl = f_gl(b, ww, data)
        if verbose > 0:
            print('[block_tseng_as] Full gradient recomputed,'
                  ' building the new active set.')
        for s in grid_x:
            ngc[s] = np.linalg.norm(full_gl(data['groups'][s]))
        if as_ is None:
            # If active set gets empty one time
            # (should be only at initialization),
            # re-start with the highest gradient.
            if first:
                as_ = np.argmax(ngc)
                first = 0
            else:
                break
        # Add to the active set the groups whose gradient norm is more than
        # lambda. This is equal to lambda in theory.
        # In practice, we use min(lambda,lagr).
        lagr = np.max(ngc[as_])
        term1 = np.where(not [np.sum(_ == as_) for _ in range(len(grid_x))])
        term2 = ngc > np.min(lambda_, lagr)
        nas_idx = np.nonzero(np.logical_and(term1, term2))
        if np.any(nas_idx):
            tmp = ngc[nas_idx]
            # max(tmp)
            ngc_argmax = np.argsort(tmp)
            ngc_argmax = ngc_argmax[:np.min(n_add, len(nas_idx))]
            as_ = list(as_)
            as_.append(nas_idx[ngc_argmax])
            s_idx = [data['groups'][g] for g in as_]
        else:
            global_sol = 1
            members = [np.sum(_ == range(len(grid_x))) for _ in as_]
            next_grad = np.max(ngc[not members])
        run = run + 1
    # Find groups which could be active in other argmin
    if len(as_) != 0:
        tmp = (ngc >= (lambda_ - np.max(np.abs(lambda_ - ngc[as_]))))
        complete_as = grid_x[tmp]
    else:
        complete_as = []
    as_p = []
    for s in grid_x:
        if (s in as_) and np.linalg.norm(ww[data['groups'][s]]):
            as_p.append(s)
    # Remove from AS the groups that vanished during optimization
    as_ = as_p
    return b, ww, as_, complete_as, next_grad


def block_tseng(f, f_h, b, w0, data, lambda_, d, options):
    """
    Laurent Jacob, Feb. 2009
    Inspired by G. Obozinski, Apr. 2008
    This code comes with no guarantee or warranty of any kind.
    This function minimizes Loss(Y,XW+B) + lambda*||W||_{1,2} using the
    algorithm of Tseng et al.
    Input: b_w0 = initial iterate
    :param f:   loss function, the calling sequence for f should be
                [fout,gout]=f(x) where fout=f(x) is a scalar and
                gout = grad f(x) is a COLUMN vector
    :param f_h: loss hessian approximation (see Tseng et al.)
                lambda is the regularization coefficient
    :param b:
    :param w0:
    :param data:
    :param lambda_:
    :param d:   is a vector of weights to change the penalty of each group.
    :param options:
    :return: BW = [B; W], where W is the vector of parameters p
            B is the set of non-regularized bias
    """
    w = w0
    groups = data['groups']
    max_it = options['max_it']
    display = options['display']
    verbose = 0
    tol = options['tol']
    tol_dj = np.sqrt(tol) / 100.
    j_old = np.inf
    s = len(groups)
    ngc = np.inf * np.ones(s + 1)
    w_norms = np.zeros(s)
    dj = np.inf * np.ones(s + 1)
    itc = 0
    while np.any(ngc > tol) and itc < max_it and np.sum(dj) > tol_dj:
        s = np.mod(itc, s + 1)
        if not s:
            f_c0, g_c0 = f(b, w, s, data, 0)
            # Recalculating the gradient now that the step has been taken
            ngc[s + 1] = np.linalg.norm(g_c0)
            hess = f_h(b, w, 0, data)
            xc = b
            b = tseng_newton(
                xc, f, b, w, 1, hess, g_c0, f_c0, 0, lambda_, data)
        else:
            f_cs, g_cs = f(b, w, s, data, 1)
            if np.linalg.norm(g_cs) <= d[s] * lambda_:
                w[groups[s]] = 0
                ngc[s + 1] = 0
            else:
                f_cs, g_cs = f(b, w, s, data, 0)
                n_ws = np.linalg.norm(w[groups[s]])
                if n_ws != 0:
                    w_nw = w[groups[s]] / n_ws
                    ngc[s + 1] = np.linalg.norm(lambda_ * d[s] * w_nw + g_cs)
                else:
                    ng_cs_curr = np.linalg.norm(g_cs)
                    if ng_cs_curr > 0:
                        tmp = g_cs * (1. - lambda_ * d[s] / ng_cs_curr)
                        ngc[s + 1] = np.linalg.norm(tmp)
                    else:
                        ngc[s + 1] = 0
                # Evaluation of the hessian at the current block
                hess = f_h(b, w, s, data)
                xc = w[groups[s]]
                # xc, f, b, w, ds, jh, jg, j_fc, s, lambda_, data
                w[groups[s]] = tseng_newton(xc, f, b, w, d[s], hess, g_cs,
                                            f_cs, s, lambda_, data)
        itc += 1
        if not s:
            j = f(b, w, 0, data, 0)
        else:
            j = f(b, w, 0, data, 0) + lambda_ * d * w_norms
        dj[s + 1] = j_old - j
        j_old = j
    if itc == max_it:
        status = 'max_iter'
    elif np.all(ngc < tol):
        status = 'gradient_tol'
    else:
        status = 'fun_tol'
    if verbose > 0:
        print('[block_tseng] Optimization ended with status '
              '%s, J=%.4f, itc=%d, l2_crit=%.4f, dj=%.4f\n',
              status, j_old, itc, np.dot(ngc, ngc), np.sum(dj))
    return b, w


def test(data, w, loss):
    """
    To do prediction on test dataset
    :param data:
    :param w:
    :param loss:
    :return:
    """
    from sklearn.metrics import roc_auc_score
    x = data['x']
    y = data['y']
    if loss == 'l2':
        y_pred = np.dot(x, w)
        diff = y_pred - y
        err = np.dot(diff, diff) / len(y)
    elif loss == 'l2Bal':
        posi_idx = np.nonzero(y < 0)
        nega_idx = np.nonzero(y > 0)
        cp = len(nega_idx) / float(len(y))
        cn = len(posi_idx) / float(len(y))
        yp = y[posi_idx]
        yn = y[nega_idx]
        y_pred = np.dot(x, w)
        y_pp = y_pred[posi_idx]
        y_pn = y_pred[nega_idx]
        diff_p = y_pp - yp
        diff_n = y_pn - yn
        err = cp * np.dot(diff_p, diff_p) + cn * np.dot(diff_n, diff_n)
    elif loss == 'class':
        y_pred = np.dot(x, w)
        err = np.sum(y != np.sign(y_pred)) / float(len(y))
    elif loss == 'classBal':
        err = None
    elif loss == 'all':
        posi_idx = np.nonzero(y > 0)
        nega_idx = np.nonzero(y < 0)
        y_pred = np.dot(x, w)
        err = dict()
        err['auc'] = roc_auc_score(y_true=y, y_score=y_pred)
        err['acc'] = np.sum(y != np.sin(y_pred)) / float(len(y))
        term1 = np.sum(np.sign(y_pred[posi_idx]) != 1)
        term1 = term1 / float(np.max(1, len(posi_idx)))
        term2 = np.sum(np.sign(y_pred[nega_idx]) != -1)
        term2 = term2 / float(np.max(1, len(nega_idx)))
        err['bacc'] = (term1 + term2) / 2.
    else:
        print('this loss is not implemented.')
        err = None
    return err


def over_lasso(data, tr_idx, te_idx, lambdas, w0, use_as, g_weight, loss):
    """
     This function optimizes a given loss function with the overlap penalty of
     (Jacob, Obozinski and Vert, 2009). In practice, given the data X, Y and
     the groups of covariates, it creates for each original variable one
     duplicate for each of its occurrences in a group, then runs regular group
     lasso on this latent representation.
    :param data:    a structure with fields 'X', 'Y' and 'groups', the latter
            being a cell array, such that data.groups{g} contains the indices
            of the variables belonging to group g.
    :param tr_idx:  arrays containing the rows of X and Y to be used as
            training examples.
    :param te_idx:  arrays containing the rows of X and Y to be used as
            testing examples.
    :param lambdas: the regularization parameter values for which the
            optimization should be performed.
    :param w0:      an initialization of [B W], or [].
    :param use_as:      1 to use the active set approach (recommended), 0
    :param g_weight:    1 to reweight the group norms in the penalty, 0
    :param loss: smooth loss function used for learning. Choose among
            squared loss ('l2'), logistic regression ('lr') and balanced
            logistic regression ('blr'). Note that the performance measure
            is automatically adapted to the type of loss.
    :return: res, a structure containing the performance for each lambda
            (perf), the predicted outputs (pred), the used lambdas (lambdas),
            the active variables at the optimum for each lambda (AS) and the
            optimal function in both the original (oWs) and expanded (Ws)
            spaces for each lambda.
    """
    # Used to run a less penalized learning round on the selected
    #  variables as a postprocessing. 0 to desactivate it.
    res = dict()
    unshrink = 0
    options = dict()
    options['display'] = 1000
    options['max_it'] = 2e4
    options['tol'] = 1e-5
    options['bias'] = 0
    e_data = expand_data(data)  # Duplication step
    tr_data = dict()
    te_data = dict()
    tr_data['x'] = e_data['x'][tr_idx, :]
    tr_data['y'] = e_data['y'][tr_idx]
    tr_data['groups'] = e_data['groups']
    te_data['x'] = np.ones(shape=(te_idx, e_data['x'].shape[1] + 1))
    te_data['x'][te_idx, 1:] = e_data['x'][te_idx, :]
    te_data['y'] = e_data['y'][te_idx]
    n, p = np.shape(tr_data['x'])
    # Select used function according to the chosen loss
    # f/fH/fGL: addresses of functions returning the loss function and
    # its (partial) gradient (f), its approximated hessian (fH) and the
    # full gradient vector (fGL). In addition, f_test selects the performance
    # measure (see test.m).
    f, f_h, f_gl = None, None, None
    if loss == 'lr':
        f_test = 'class'
    elif loss == 'blr':
        np_ = np.sum(tr_data['y'] == 1)
        nn_ = np.sum(tr_data['y'] == -1)
        n = len(tr_data['y'])
        cp = nn_ / n
        cn = np_ / n
        f = lambda b_, w_, s_, data_, at_zero_: \
            blr_oracle(b_, w_, s_, data_, at_zero_, cp, cn)
        f_h = lambda b_, w_, s_, data_: blr_h(b_, w_, s_, data_, cp, cn)
        f_gl = lambda b_, w_, data_: blr_gl(b_, w_, data_, cp, cn)
        f_test = 'all'  # return accuracy, balanced accuracy and AUC.
    elif loss == 'l2':
        f_test = 'l2'
    else:
        f_test = 'None'
    d = np.ones(len(tr_data['groups']))
    if g_weight:
        for g in range(len(tr_data['groups'])):
            svd_x = np.linalg.svd(tr_data['x'][:, tr_data['groups'][g]])
            d[g] = np.sum(svd_x)
        d = d / max(d)
    ws = np.zeros((p + 1, len(lambdas)))
    as_arr = {i: dict() for i in range(len(lambdas))}
    complete_as_arr = {i: dict() for i in range(len(lambdas))}
    if w0 is not None:
        bw = w0
    else:
        bw = np.zeros(p + 1)
    if not f_test == 'all':
        res['perf'] = -np.ones(len(lambdas))
    else:
        res['auc'] = -np.ones(len(lambdas))
        res['acc'] = -np.ones(len(lambdas))
        res['bacc'] = -np.ones(len(lambdas))
    res['pred'] = np.zeros((len(te_data['x']), len(lambdas)))
    res['lambdas'] = lambdas
    res['nextGrad'] = -np.ones(len(lambdas))
    if use_as:
        as_ = []
    else:
        as_ = range(len(tr_data['groups']))
    options['max_feat'] = np.shape(tr_data['x'])[1]
    # For each lambda on the grid, calls the optimization function (with warm
    # restart, i.e., starting from the previous optimal function/active set).
    i = 0
    for lambda_ in lambdas:
        i += 1
        print('[overLasso] Starting optimization with lambda=%.4f' % lambda_)
        para = block_tseng_as(
            f, f_h, f_gl, bw, as_, tr_data, lambda_, d, options)
        b, w, as_, complete_as, next_grad = para
        if not use_as:
            as_ = range(len(tr_data['groups']))
            complete_as = as_
        # s_idx must contain the indices of the variables
        # from all the groups in the active set.
        if unshrink:
            s_idx = [data['groups'][g] for g in as_]
            sub_data = dict()
            sub_data['x'] = tr_data['x'][:, s_idx]
            sub_data['y'] = tr_data['y']
            sub_data['groups'] = {i: dict() for i in range(len(as_))}
            c = 0
            for g in as_:
                c += 1
                tmp = is_member(s_idx, tr_data['groups'][g])
                sub_data['groups'][c] = np.nonzero(tmp)
            sbw = block_tseng(f, f_h, bw[0], bw[s_idx], sub_data, unshrink,
                              d[as_], options)
            ws[s_idx, i] = sbw
        else:
            ws[:, i] = bw
        res['nextGrad'][i] = next_grad
        as_arr[i] = as_
        complete_as_arr[i] = complete_as
        if len(te_idx) != 0:
            if f_test != 'all':
                re_1, re_2 = test(te_data, ws[:, i], f_test)
                res['perf'][i] = re_1
                res['pred'][:, i] = re_2
            else:
                re_1, re_2 = test(te_data, ws[:, i], f_test)
                res['auc'][i] = re_1['auc']
                res['acc'][i] = re_1['acc']
                res['bacc'][i] = re_1['bacc']
                res['pred'][:, i] = re_2
    if f_test == 'all':
        res['perf'] = res['bacc']
    res['as'] = as_arr
    res['complete_as'] = complete_as_arr
    res['ws'] = ws
    res['o_ws'] = [ws[0, :], e_data['exp_2_orig'] * ws[1:, :]]
    return res


def cv_over_lasso(data, splits_str, lambdas, proc_type, loss):
    splits = splits_str['splits']
    sub_splits = splits_str['sub_splits']
    n_folds = len(splits)
    ns_folds = len(splits[0])
    cv_res = {i: dict() for i in range(n_folds)}
    g_weight = 0
    use_as = 1
    for i_fold in range(n_folds):
        print('[cvOverLasso] Learning on fold %02d' % i_fold)
        tr_idx = splits[i_fold]['x_tr']
        te_idx = splits[i_fold]['x_te']
        f_data = data
        # split data
        tr_data = dict()
        tr_data['x'] = f_data['x'][tr_idx, :]
        tr_data['y'] = f_data['y'][tr_idx]
        tr_data['entrez'] = f_data['entrez']
        tr_data['groups'] = tr_data['groups']
        # process based only on training data
        tr_data, k_idx, k_groups, k_groups_idx = process(tr_data, proc_type)
        f_data['x'] = f_data['x'][:, k_idx]
        f_data['entrez'] = tr_data['entrez']
        f_data['groups'] = tr_data['groups']

        cv_res[i_fold]['lambdas'] = lambdas
        cv_res[i_fold]['k_idx'] = k_idx
        cv_res[i_fold]['k_groups'] = k_groups
        cv_res[i_fold]['k_groups_idx'] = k_groups_idx
        cv_res[i_fold]['groups'] = f_data['groups']

        if loss == 'blr':
            s_auc = np.zeros(len(lambdas))
            s_acc = np.zeros(len(lambdas))
            s_b_acc = np.zeros(len(lambdas))
            s_perfs = np.zeros(len(lambdas))
        else:
            s_auc = np.zeros(len(lambdas))
            s_acc = np.zeros(len(lambdas))
            s_b_acc = np.zeros(len(lambdas))
            s_perfs = np.zeros(len(lambdas))
        l_res = dict()
        for is_fold in range(ns_folds):
            print('[cv_over_lasso] In internal CV, '
                  'learning on subfold %d' % is_fold)
            s_x_tr = sub_splits[i_fold][is_fold]['x_tr']
            s_x_te = sub_splits[i_fold][is_fold]['x_te']
            l_res[is_fold] = over_lasso(
                f_data, s_x_tr, s_x_te, lambdas, [], use_as, g_weight, loss)
            if loss == 'blr':
                s_auc = s_auc + l_res[is_fold]['auc'] / n_folds
                s_acc = s_acc + l_res[is_fold]['acc'] / n_folds
                s_b_acc = s_b_acc + l_res[is_fold]['bacc'] / n_folds
            else:
                s_perfs = s_perfs + l_res[is_fold]['perf'] / n_folds
        if loss == 'blr':
            l_star = np.argmin(s_b_acc)
            cv_res[i_fold]['s_b_acc'] = s_b_acc
        else:
            l_star = np.argmin(s_perfs)
        res = over_lasso(f_data, tr_idx, te_idx, lambdas, [], use_as,
                         g_weight, loss)
        cv_res[i_fold]['as'] = res['as']
        cv_res[i_fold]['complete_as'] = res['complete_as']
        cv_res[i_fold]['l_star'] = lambdas['l_star']
        if loss == 'blr':
            cv_res[i_fold]['auc'] = res['auc']
            cv_res[i_fold]['acc'] = res['acc']
            cv_res[i_fold]['b_acc'] = res['bacc']
            cv_res[i_fold]['perf'] = res['bacc'][l_star]
        else:
            cv_res[i_fold]['perf'] = res['perf']
        cv_res[i_fold]['pred'] = res['pred']
        cv_res[i_fold]['ws'] = res['ws']
        cv_res[i_fold]['o_ws'] = res['o_ws']
        cv_res[i_fold]['next_grad'] = res['next_grad']
    return cv_res


def edges_to_groups(edges, k):
    """
    :param edges: is a p*2 matrix of pairwise relationships (edges)
    :param k: is the size of the groups we want
    :return:
    """
    n_edges = np.shape(edges)[0]
    n_vertices = np.max(edges.flatten())
    groups = dict()
    for i in range(n_vertices):
        groups[i] = i
    if k > 1:
        for i in range(n_edges):
            groups[n_vertices + i] = edges[i, :]
    return groups


def main_bc_graph():
    dataset = 'vv'  # dataset
    loss = 'blr'  # chosen from [l2, lr, blr]
    preproc = 'correl'  # plain or correl
    dparam = 0  #
    nfolds = [3, 3]  # number of folds
    data, splits_str = get_data(dataset, nfolds, dparam)
    nodes, edges, costs = set(), [], []
    with open(root_p + 'raw/edge.txt') as csvfile:
        edge_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in edge_reader:
            uu, vv = int(row[0]) - 1, int(row[1]) - 1
            nodes.add(uu)
            nodes.add(vv)
            edges.append([uu, vv])
            costs.append(1.)
    print('number of nodes: %d' % len(nodes))
    print(len(data['x'][0]))
    data['x'] = data['x'][:, list(nodes)]
    data['entrez'] = data['entrez'][list(nodes)]
    lambdas = 2. ** np.asarray(range(-8, 1, 2))[::-1]
    # lasso penalty
    print('main_bcgr starting lasso')
    data['groups'] = {i: [i] for i in range(len(data['x'][0]))}
    res = dict()
    res[0] = cv_over_lasso(data, splits_str, lambdas, preproc, loss)
    # Groups are the edges
    print('main_bcgr starting structure lasso on edges')
    data['groups'] = edges_to_groups(np.asarray(edges), 2)
    res[1] = cv_over_lasso(data, splits_str, lambdas, preproc, loss)
    print(len(data['x'][0]))
    print(max(nodes), min(nodes))
    print(lambdas)


def main_bc_path():
    dataset = 'vv'  # dataset
    loss = 'blr'  # chosen from [l2, lr, blr]
    preproc = 'correl'  # plain or correl
    dparam = 0  #
    nfolds = [3, 3]  # number of folds
    data, splits_str = get_data(dataset, nfolds, dparam)
    print('[main_bc_path] Data formatted')
    pathways, all_nodes = dict(), set()
    with open(root_p + 'raw/pathways.txt') as csvfile:
        pathways_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in pathways_reader:
            pathway_id, pathway_node = int(row[0]) - 1, int(row[1]) - 1
            all_nodes.add(pathway_node)
            if pathway_id not in pathways:
                pathways[pathway_id] = set()
            pathways[pathway_id].add(pathway_node)
    print(len(data['x'][0]))
    # Keep only genes that are in at least one pathway
    tmp = is_member(data['entrez'], np.unique(all_nodes))
    var_in_path = np.nonzero(tmp)
    data['x'] = data['x'][:, list(var_in_path)]
    data['entrez'] = data['entrez'][list(var_in_path)]
    grids = np.unique(pathways.keys())
    p_groups = {i: dict() for i in range(len(grids))}
    c = 0
    for g in grids:
        c += 1
        c_entrez = pathways[g]
        tmp = is_member(data['entrez'], c_entrez)
        p_groups[c] = np.nonzero(tmp)
    res = dict()
    lambdas = 2. ** np.asarray(np.arange(-12, 0.01, 0.1))[::-1]
    print('main_bc_path starting structure lasso')
    res[0] = cv_over_lasso(data, splits_str, lambdas, preproc, loss)
    # Groups are the pathways
    print('[main_bc_group] Starting structure lasso on pathways')
    data['groups'] = p_groups
    res[1] = cv_over_lasso(data, splits_str, lambdas, preproc, loss)
    print(len(data['x'][0]))


def process_single_run(trial_i):
    import scipy.io as sio
    results = dict()
    f_name = 'results_bc_path_overlap_lasso_%02d.mat' % trial_i
    if os.path.exists(root_output + f_name):
        bc_path = sio.loadmat(root_output + f_name)['ans']
    else:
        print('error')
        exit(0)
    results['path_lasso'] = dict()
    results['path_overlap_lasso'] = dict()
    results['graph_lasso'] = dict()
    results['graph_overlap_lasso'] = dict()
    for re_ind, re in enumerate(bc_path[0][0][0]):
        results['path_lasso']['folding_%d' % (re_ind + 1)] = re[0][0]
    for re_ind, re in enumerate(bc_path[0][1][0]):
        results['path_overlap_lasso']['folding_%d' % (re_ind + 1)] = re[0][0]
    f_name = 'results_bc_path_overlap_lasso_01.mat'
    bc_graph = sio.loadmat(root_output + f_name)['ans']
    for re_ind, re in enumerate(bc_graph[0][0][0]):
        results['graph_lasso']['folding_%d' % (re_ind + 1)] = re[0][0]
    for re_ind, re in enumerate(bc_graph[0][1][0]):
        results['graph_overlap_lasso']['folding_%d' % (re_ind + 1)] = re[0][0]
    keys_dict = ['lambdas', 'kidx', 'kgroups', 'kgroupidx', 'groups', 'sbacc',
                 'AS', 'completeAS', 'lstar', 'auc', 'acc', 'bacc', 'perf',
                 'pred', 'Ws', 'oWs', 'nextGrad']
    fold_list = ['folding_1', 'folding_2', 'folding_3',
                 'folding_4', 'folding_5']
    summary_results = dict()
    for method in ['path_lasso', 'path_overlap_lasso', 'graph_lasso',
                   'graph_overlap_lasso']:
        summary_results[method] = dict()
        for fold_i in fold_list:
            summary_results[method][fold_i] = dict()
            for key in keys_dict:
                re = results[method][fold_i][key][0]
                summary_results[method][fold_i][key] = re
    print('_' * 60)
    print('-' * 60)
    for fold_i in fold_list:
        re = summary_results['path_overlap_lasso'][fold_i]['bacc']
        print('Error %s:    %.2f(%.2f)' %
              (fold_i, float(np.mean(re)), float(np.std(re)))),
        summary_results['path_overlap_lasso'][fold_i]['bacc'] = np.min(re)
        re = summary_results['path_lasso'][fold_i]['bacc']
        print('    %.2f(%.2f)' %
              (float(np.mean(re)), float(np.std(re))))
        summary_results['path_lasso'][fold_i]['bacc'] = np.min(re)
    return summary_results


def process_single_run2(trial_i, method_flag):
    import scipy.io as sio
    results = dict()
    f_name = 'results_bc_%s_overlap_lasso_%02d.mat' % (method_flag, trial_i)
    bc_path = sio.loadmat(root_output + f_name)['ans']
    method1 = '%s_lasso' % method_flag
    method2 = '%s_overlap_lasso' % method_flag
    results[method1] = dict()
    results[method2] = dict()
    for re_ind, re in enumerate(bc_path[0][0][0]):
        results[method1]['folding_%d' % (re_ind + 1)] = re[0][0]
    for re_ind, re in enumerate(bc_path[0][1][0]):
        results[method2]['folding_%d' % (re_ind + 1)] = re[0][0]
    keys_dict = ['lambdas', 'kidx', 'kgroups', 'kgroupidx', 'groups', 'sbacc',
                 'AS', 'completeAS', 'lstar', 'auc', 'acc', 'bacc', 'perf',
                 'pred', 'Ws', 'oWs', 'nextGrad']
    fold_list = ['folding_%d' % i for i in range(1, 6)]
    summary_results = dict()
    for method in [method1, method2]:
        summary_results[method] = dict()
        for fold_i in fold_list:
            summary_results[method][fold_i] = dict()
            for key in keys_dict:
                re = results[method][fold_i][key][0]
                summary_results[method][fold_i][key] = re
    bacc_lasso = np.zeros((5, 121))
    bacc_overlap_lasso = np.zeros((5, 121))
    for fold_ind, fold_i in enumerate(fold_list):
        re = summary_results[method1][fold_i]['perf']
        bacc_lasso[fold_ind] = re
        re = summary_results[method2][fold_i]['perf']
        bacc_overlap_lasso[fold_ind] = re
    print(np.mean(bacc_lasso, axis=0))
    best_para = np.argmin(np.mean(bacc_lasso, axis=0))
    best_bacc_lasso = bacc_lasso[:, best_para]
    best_para = np.argmin(np.mean(bacc_overlap_lasso, axis=0))
    best_bacc_overlap_lasso = bacc_overlap_lasso[:, best_para]
    return best_bacc_lasso, best_bacc_overlap_lasso


def result_analysis(method='path'):
    fold_list = ['folding_1', 'folding_2', 'folding_3',
                 'folding_4', 'folding_5']
    all_bacc_lasso = np.zeros((10, 5))
    all_bacc_overlap_lasso = np.zeros((10, 5))
    for trial_i in range(1, 11):
        bacc_lasso, bacc_overlap_lasso = process_single_run2(trial_i, method)
        all_bacc_lasso[trial_i - 1] = bacc_lasso
        all_bacc_overlap_lasso[trial_i - 1] = bacc_overlap_lasso
    print('-' * 80)
    for fold_ind, fold_i in enumerate(fold_list):
        re = all_bacc_overlap_lasso[:, fold_ind]
        print('Error %s:    %.4f(%.4f)' %
              (fold_i, float(np.mean(re)), float(np.std(re)))),
        re = all_bacc_lasso[:, fold_ind]
        print('    %.4f(%.4f)' %
              (float(np.mean(re)), float(np.std(re))))


def main():
    command = sys.argv[1]
    if command == 'run_test01':
        main_bc_graph()
    elif command == 'run_test02':
        main_bc_path()
    elif command == 'show_test01':
        result_analysis()


if __name__ == '__main__':
    main()
