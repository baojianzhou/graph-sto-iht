# -*- coding: utf-8 -*-
import os
import csv
import sys
import pickle
import numpy as np
import multiprocessing
from itertools import product
from numpy.random import randint

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


def expit(x):
    """
    expit function. 1 /(1+exp(-x)). quote from Scipy:
    The expit function, also known as the logistic function,
    is defined as expit(x) = 1/(1+exp(-x)).
    It is the inverse of the logit function.
    expit is also known as logistic. Please see logistic
    :param x: np.ndarray
    :return: 1/(1+exp(-x)).
    """
    out = np.zeros_like(x)
    posi = np.where(x > 0.0)
    nega = np.where(x <= 0.0)
    out[posi] = 1. / (1. + np.exp(-x[posi]))
    exp_x = np.exp(x[nega])
    out[nega] = exp_x / (1. + exp_x)
    return out


def logistic_predict(x, wt):
    """
    To predict the probability for sample xi. {+1,-1}
    :param x: (n,p) dimension, where p is the number of features.
    :param wt: (p+1,) dimension, where wt[p] is the intercept.
    :return: (n,1) dimension of predict probability of positive class
            and labels.
    """
    n, p = x.shape
    pred_prob = expit(np.dot(x, wt[:p]) + wt[p])
    pred_y = np.ones(n)
    pred_y[pred_prob < 0.5] = -1.
    return pred_prob, pred_y


def log_logistic(x):
    """ return log( 1/(1+exp(-x)) )"""
    out = np.zeros_like(x)
    posi = np.where(x > 0.0)
    nega = np.where(x <= 0.0)
    out[posi] = -np.log(1. + np.exp(-x[posi]))
    out[nega] = x[nega] - np.log(1. + np.exp(x[nega]))
    return out


def logit_loss_grad_bl(x_tr, y_tr, wt, l2_reg, cp, cn):
    """
    Calculate the balanced loss and gradient of the logistic function.
    :param x_tr: (n,p), where p is the number of features.
    :param y_tr: (n,), where n is the number of labels.
    :param wt: current model. wt[-1] is the intercept.
    :param l2_reg: regularization to avoid overfitting.
    :param cp:
    :param cn:
    :return: {+1,-1} Logistic (val,grad) on training samples.
    """
    assert len(wt) == (x_tr.shape[1] + 1)
    c, n, p = wt[-1], x_tr.shape[0], x_tr.shape[1]
    posi_idx = np.where(y_tr > 0)  # corresponding to positive labels.
    nega_idx = np.where(y_tr < 0)  # corresponding to negative labels.
    grad = np.zeros_like(wt)
    wt = wt[:p]
    yz = y_tr * (np.dot(x_tr, wt) + c)
    z = expit(yz)
    loss = -cp * np.sum(log_logistic(yz[posi_idx]))
    loss += -cn * np.sum(log_logistic(yz[nega_idx]))
    loss = loss / n + .5 * l2_reg * np.dot(wt, wt)
    bl_y_tr = np.zeros_like(y_tr)
    bl_y_tr[posi_idx] = cp * np.asarray(y_tr[posi_idx], dtype=float)
    bl_y_tr[nega_idx] = cn * np.asarray(y_tr[nega_idx], dtype=float)
    z0 = (z - 1) * bl_y_tr  # z0 = (z - 1) * y_tr
    grad[:p] = np.dot(x_tr.T, z0) / n + l2_reg * wt
    grad[-1] = z0.sum()  # do not need to regularize the intercept.
    return loss, grad


def logit_loss_bl(x_tr, y_tr, wt, l2_reg, cp, cn):
    """
    Calculate the balanced loss and gradient of the logistic function.
    :param x_tr: (n,p), where p is the number of features.
    :param y_tr: (n,), where n is the number of labels.
    :param wt: current model. wt[-1] is the intercept.
    :param l2_reg: regularization to avoid overfitting.
    :param cp:
    :param cn:
    :return: return {+1,-1} Logistic (val,grad) on training samples.
    """
    assert len(wt) == (x_tr.shape[1] + 1)
    c, n, p = wt[-1], x_tr.shape[0], x_tr.shape[1]
    posi_idx = np.where(y_tr > 0)  # corresponding to positive labels.
    nega_idx = np.where(y_tr < 0)  # corresponding to negative labels.
    wt = wt[:p]
    yz = y_tr * (np.dot(x_tr, wt) + c)
    loss = -cp * np.sum(log_logistic(yz[posi_idx]))
    loss += -cn * np.sum(log_logistic(yz[nega_idx]))
    loss = loss / n + .5 * l2_reg * np.dot(wt, wt)
    return loss


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


def algo_graph_sto_iht_backtracking(
        x_tr, y_tr, w0, max_epochs, s, edges, costs, num_blocks, lambda_,
        g=1, root=-1, gamma=0.1, proj_max_num_iter=50, verbose=0):
    np.random.seed()  # do not forget it.
    w_hat = np.copy(w0)
    (m, p) = x_tr.shape
    # if the block size is too large. just use single block
    b = int(m) / int(num_blocks)
    np_ = np.sum(y_tr == 1)
    nn_ = np.sum(y_tr == -1)
    cp = float(nn_) / float(len(y_tr))
    cn = float(np_) / float(len(y_tr))

    # graph projection para
    h_low = int(2 * s)
    h_high = int(2 * (1. + gamma))
    t_low = int(s)
    t_high = int(s * (1. + gamma))
    for epoch_i in range(max_epochs):
        for ind, _ in enumerate(range(num_blocks)):
            ii = randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            x_tr_b, y_tr_b = x_tr[block, :], y_tr[block]
            loss_sto, grad_sto = logit_loss_grad_bl(
                x_tr=x_tr_b, y_tr=y_tr_b, wt=w_hat,
                l2_reg=lambda_, cp=cp, cn=cn)
            h_nodes, p_grad = algo_head_tail_bisearch(
                edges, grad_sto[:p], costs, g, root, h_low, h_high,
                proj_max_num_iter, verbose)
            p_grad = np.append(p_grad, grad_sto[-1])
            fun_val_right = loss_sto
            tmp_num_iter, ad_step, beta = 0, 1.0, 0.8
            reg_term = np.linalg.norm(p_grad) ** 2.
            while tmp_num_iter < 20:
                x_tmp = w_hat - ad_step * p_grad
                fun_val_left = logit_loss_bl(
                    x_tr=x_tr_b, y_tr=y_tr_b, wt=x_tmp,
                    l2_reg=lambda_, cp=cp, cn=cn)
                if fun_val_left > fun_val_right - ad_step / 2. * reg_term:
                    ad_step *= beta
                else:
                    break
                tmp_num_iter += 1
            bt_sto = np.zeros_like(w_hat)
            bt_sto[:p] = w_hat[:p] - ad_step * p_grad[:p]
            t_nodes, proj_bt = algo_head_tail_bisearch(
                edges, bt_sto[:p], costs, g, root, t_low, t_high,
                proj_max_num_iter, verbose)
            w_hat[:p] = proj_bt[:p]
            w_hat[p] = w_hat[p] - ad_step * grad_sto[p]  # intercept.
    return w_hat


def algo_sto_iht_backtracking(
        x_tr, y_tr, w0, max_epochs, s, num_blocks, lambda_):
    np.random.seed()  # do not forget it.
    w_hat = w0
    (m, p) = x_tr.shape
    b = int(m) / int(num_blocks)
    np_ = np.sum(y_tr == 1)
    nn_ = np.sum(y_tr == -1)
    cp = float(nn_) / float(len(y_tr))
    cn = float(np_) / float(len(y_tr))
    for epoch_i in range(max_epochs):
        for ind, _ in enumerate(range(num_blocks)):
            ii = randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            x_tr_b, y_tr_b = x_tr[block, :], y_tr[block]
            loss_sto, grad_sto = logit_loss_grad_bl(
                x_tr=x_tr_b, y_tr=y_tr_b, wt=w_hat,
                l2_reg=lambda_, cp=cp, cn=cn)
            fun_val_right = loss_sto
            tmp_num_iter, ad_step, beta = 0, 1.0, 0.8
            reg_term = np.linalg.norm(grad_sto) ** 2.
            while tmp_num_iter < 20:
                x_tmp = w_hat - ad_step * grad_sto
                fun_val_left = logit_loss_bl(
                    x_tr=x_tr_b, y_tr=y_tr_b, wt=x_tmp,
                    l2_reg=lambda_, cp=cp, cn=cn)
                if fun_val_left > fun_val_right - ad_step / 2. * reg_term:
                    ad_step *= beta
                else:
                    break
                tmp_num_iter += 1
            bt_sto = w_hat - ad_step * grad_sto
            bt_sto[np.argsort(np.abs(bt_sto))[:p - s]] = 0.
            w_hat = bt_sto
    return w_hat


def cv_split_bal(y, n_folds):
    splits = {i: dict() for i in range(n_folds)}
    posi_idx = np.nonzero(y == 1)[0]
    nega_idx = np.nonzero(y == -1)[0]
    np_, nn_ = len(posi_idx), len(nega_idx)
    p_size_fold = int(np.round(float(len(posi_idx)) / float(n_folds)))
    n_size_fold = int(np.round(float(len(nega_idx)) / float(n_folds)))
    pa = posi_idx[np.random.permutation(len(posi_idx))]
    na = nega_idx[np.random.permutation(len(nega_idx))]
    # to split all training samples into n_folds
    for i in range(n_folds):
        if i < n_folds - 1:
            posi = pa[0:i * p_size_fold]
            posi = np.append(posi, pa[(i + 1) * p_size_fold:np_])
            nega = na[0:i * n_size_fold]
            nega = np.append(nega, na[(i + 1) * n_size_fold:nn_])
            splits[i]['x_tr'] = np.append(posi, nega)
            # leave one folder
            posi = pa[i * p_size_fold:(i + 1) * p_size_fold]
            nega = na[i * n_size_fold:(i + 1) * n_size_fold]
            splits[i]['x_te'] = np.append(posi, nega)
            pass
        else:
            posi = pa[0:i * p_size_fold]
            nega = na[0:i * n_size_fold]
            splits[i]['x_tr'] = np.append(posi, nega)
            posi = pa[i * p_size_fold:np_]
            nega = na[i * n_size_fold:nn_]
            splits[i]['x_te'] = np.append(posi, nega)
    return splits


def process(data, edges):
    p_data = dict()
    p = data['x'].shape[1]
    r = np.zeros(p)
    for i in range(p):
        c = np.corrcoef(data['x'][:, i], data['y'])
        r[i] = c[0][1]
    k_idx = np.argsort(np.abs(r))[-500:]

    import networkx as nx
    g = nx.Graph()
    for edge in edges:
        g.add_edge(u_of_edge=edge[0], v_of_edge=edge[1])
    print('number of nodes: %d' % nx.number_of_nodes(g))
    print('number of edges: %d' % nx.number_of_edges(g))
    print('number of cc: %d' % nx.number_connected_components(g))
    max_cc, max_cc_idx = None, -1
    for ind, cc in enumerate(nx.connected_component_subgraphs(g)):
        if max_cc is None or len(max_cc) < len(cc):
            max_cc = {i: '' for i in list(cc)}
    print('about %02d nodes which are not in top 500 but removed' %
          len([k for k in k_idx if k not in max_cc]))
    ratio = float(len([k for k in k_idx if k not in max_cc])) / 500.
    print('the ratio that nodes not in top 500 is about: %.4f' % ratio)
    reduced_edges, reduced_nodes = [], []
    red_g = nx.Graph()
    for edge in edges:
        if edge[0] in max_cc:
            reduced_edges.append(edge)
            if edge[0] not in reduced_nodes:
                reduced_nodes.append(edge[0])
            if edge[1] not in reduced_nodes:
                reduced_nodes.append(edge[1])
            red_g.add_edge(u_of_edge=edge[0], v_of_edge=edge[1])
    print('number of nodes: %d' % nx.number_of_nodes(red_g))
    print('number of edges: %d' % nx.number_of_edges(red_g))
    print('number of cc: %d' % nx.number_connected_components(red_g))

    node_ind_dict = dict()
    ind_node_dict = dict()
    for ind, node in enumerate(reduced_nodes):
        node_ind_dict[node] = ind
        ind_node_dict[ind] = node
    p_data['y'] = data['y']
    p_data['x'] = data['x'][:, reduced_nodes]
    edges, costs = [], []
    g = nx.Graph()
    for edge in reduced_edges:
        if edge[0] == edge[1]:
            print('remove self loops')
            continue
        edges.append([node_ind_dict[edge[0]], node_ind_dict[edge[1]]])
        g.add_edge(u_of_edge=edges[-1][0], v_of_edge=edges[-1][1])
        costs.append(1.)
    edges = np.asarray(edges, dtype=int)
    costs = np.asarray(costs, dtype=np.float64)
    p_data['edges'] = edges
    p_data['costs'] = costs
    print('number of nodes: %d' % nx.number_of_nodes(red_g))
    print('number of edges: %d' % nx.number_of_edges(red_g))
    print('number of cc: %d' % nx.number_connected_components(red_g))
    return p_data, k_idx, list(nx.nodes(g))


def test(data, w):
    """
    To do prediction on test dataset
    :param data:
    :param w:
    :return:
    """
    from sklearn.metrics import roc_auc_score
    x = data['x']
    y = data['y']
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
    return err


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


def run_single_test(para):
    data, tr_idx, te_idx, s, num_blocks, lambda_, \
    max_epochs, fold_i, subfold_i = para
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import accuracy_score
    res = {'iht': dict(),
           'graph-iht': dict(),
           'sto-iht': dict(),
           'graph-sto-iht': dict()}
    tr_data = dict()
    tr_data['x'] = data['x'][tr_idx, :]
    tr_data['y'] = data['y'][tr_idx]
    te_data = dict()
    te_data['x'] = data['x'][te_idx, :]
    te_data['y'] = data['y'][te_idx]
    x_tr, y_tr = tr_data['x'], tr_data['y']
    w0 = np.zeros(np.shape(x_tr)[1] + 1)
    # --------------------------------
    # this corresponding (b=1) to IHT
    w_hat = algo_sto_iht_backtracking(
        x_tr, y_tr, w0, max_epochs, s, 1, lambda_)
    x_te, y_te = te_data['x'], te_data['y']
    pred_prob, pred_y = logistic_predict(x_te, w_hat)
    posi_idx = np.nonzero(y_te == 1)[0]
    nega_idx = np.nonzero(y_te == -1)[0]
    print('-' * 80)
    print('number of positive: %02d, missed: %02d '
          'number of negative: %02d, missed: %02d ' %
          (len(posi_idx), float(np.sum(pred_y[posi_idx] != 1)),
           len(nega_idx), float(np.sum(pred_y[nega_idx] != -1))))
    v1 = np.sum(pred_y[posi_idx] != 1) / float(len(posi_idx))
    v2 = np.sum(pred_y[nega_idx] != -1) / float(len(nega_idx))
    res['iht']['bacc'] = (v1 + v2) / 2.
    res['iht']['acc'] = accuracy_score(y_true=y_te, y_pred=pred_y)
    res['iht']['auc'] = roc_auc_score(y_true=y_te, y_score=pred_prob)
    res['iht']['perf'] = res['iht']['bacc']
    res['iht']['w_hat'] = w_hat
    print('iht   -- sparsity: %02d intercept: %.4f bacc: %.4f' %
          (s, w_hat[-1], res['iht']['bacc']))
    # --------------------------------
    w_hat = algo_sto_iht_backtracking(
        x_tr, y_tr, w0, max_epochs, s, num_blocks, lambda_)
    x_te, y_te = te_data['x'], te_data['y']
    pred_prob, pred_y = logistic_predict(x_te, w_hat)
    posi_idx = np.nonzero(y_te == 1)[0]
    nega_idx = np.nonzero(y_te == -1)[0]
    print('-' * 80)
    print('number of positive: %02d, missed: %02d '
          'number of negative: %02d, missed: %02d ' %
          (len(posi_idx), float(np.sum(pred_y[posi_idx] != 1)),
           len(nega_idx), float(np.sum(pred_y[nega_idx] != -1))))
    v1 = np.sum(pred_y[posi_idx] != 1) / float(len(posi_idx))
    v2 = np.sum(pred_y[nega_idx] != -1) / float(len(nega_idx))
    res['sto-iht']['bacc'] = (v1 + v2) / 2.
    res['sto-iht']['acc'] = accuracy_score(y_true=y_te, y_pred=pred_y)
    res['sto-iht']['auc'] = roc_auc_score(y_true=y_te, y_score=pred_prob)
    res['sto-iht']['perf'] = res['sto-iht']['bacc']
    res['sto-iht']['w_hat'] = w_hat
    print('sto-iht   -- sparsity: %02d intercept: %.4f bacc: %.4f' %
          (s, w_hat[-1], res['sto-iht']['bacc']))

    tr_data = dict()
    tr_data['x'] = data['x'][tr_idx, :]
    tr_data['y'] = data['y'][tr_idx]
    te_data = dict()
    te_data['x'] = data['x'][te_idx, :]
    te_data['y'] = data['y'][te_idx]
    x_tr, y_tr = tr_data['x'], tr_data['y']
    w0 = np.zeros(np.shape(x_tr)[1] + 1)
    # --------------------------------
    # this corresponding (b=1) to GraphIHT
    w_hat = algo_graph_sto_iht_backtracking(
        x_tr, y_tr, w0, max_epochs, s,
        data['edges'], data['costs'], 1, lambda_)
    x_te, y_te = te_data['x'], te_data['y']
    pred_prob, pred_y = logistic_predict(x_te, w_hat)
    posi_idx = np.nonzero(y_te == 1)[0]
    nega_idx = np.nonzero(y_te == -1)[0]
    print('-' * 80)
    print('number of positive: %02d, missed: %02d '
          'number of negative: %02d, missed: %02d ' %
          (len(posi_idx), float(np.sum(pred_y[posi_idx] != 1)),
           len(nega_idx), float(np.sum(pred_y[nega_idx] != -1))))
    v1 = np.sum(pred_y[posi_idx] != 1) / float(len(posi_idx))
    v2 = np.sum(pred_y[nega_idx] != -1) / float(len(nega_idx))
    res['graph-iht']['bacc'] = (v1 + v2) / 2.
    res['graph-iht']['acc'] = accuracy_score(y_true=y_te, y_pred=pred_y)
    res['graph-iht']['auc'] = roc_auc_score(y_true=y_te, y_score=pred_prob)
    res['graph-iht']['perf'] = res['graph-iht']['bacc']
    res['graph-iht']['w_hat'] = w_hat
    print('graph-iht -- sparsity: %02d intercept: %.4f bacc: %.4f' %
          (s, w_hat[-1], res['graph-iht']['bacc']))

    # --------------------------------
    w_hat = algo_graph_sto_iht_backtracking(
        x_tr, y_tr, w0, max_epochs, s,
        data['edges'], data['costs'], num_blocks, lambda_)
    x_te, y_te = te_data['x'], te_data['y']
    pred_prob, pred_y = logistic_predict(x_te, w_hat)
    posi_idx = np.nonzero(y_te == 1)[0]
    nega_idx = np.nonzero(y_te == -1)[0]
    print('-' * 80)
    print('number of positive: %02d, missed: %02d '
          'number of negative: %02d, missed: %02d ' %
          (len(posi_idx), float(np.sum(pred_y[posi_idx] != 1)),
           len(nega_idx), float(np.sum(pred_y[nega_idx] != -1))))
    v1 = np.sum(pred_y[posi_idx] != 1) / float(len(posi_idx))
    v2 = np.sum(pred_y[nega_idx] != -1) / float(len(nega_idx))
    res['graph-sto-iht']['bacc'] = (v1 + v2) / 2.
    res['graph-sto-iht']['acc'] = accuracy_score(y_true=y_te, y_pred=pred_y)
    res['graph-sto-iht']['auc'] = roc_auc_score(y_true=y_te, y_score=pred_prob)
    res['graph-sto-iht']['perf'] = res['graph-sto-iht']['bacc']
    res['graph-sto-iht']['w_hat'] = w_hat
    print('graph-sto-iht -- sparsity: %02d intercept: %.4f bacc: %.4f' %
          (s, w_hat[-1], res['graph-sto-iht']['bacc']))
    return s, num_blocks, lambda_, res, fold_i, subfold_i


def run_parallel_tr(
        data, s_list, b_list, lambda_list, max_epochs, num_cpus, fold_i):
    # 5-fold cross validation
    method_list = ['iht', 'sto-iht', 'graph-iht', 'graph-sto-iht']
    s_auc = {_: {(s, num_blocks, lambda_): 0.0
                 for (s, num_blocks, lambda_) in
                 product(s_list, b_list, lambda_list)} for _ in method_list}
    s_acc = {_: {(s, num_blocks, lambda_): 0.0
                 for (s, num_blocks, lambda_) in
                 product(s_list, b_list, lambda_list)} for _ in method_list}
    s_bacc = {_: {(s, num_blocks, lambda_): 0.0
                  for (s, num_blocks, lambda_) in
                  product(s_list, b_list, lambda_list)} for _ in method_list}
    input_paras = []
    for sf_ii in range(len(data['data_subsplits'][fold_i])):
        s_tr = data['data_subsplits'][fold_i][sf_ii]['train']
        s_te = data['data_subsplits'][fold_i][sf_ii]['test']
        for s, num_block, lambda_ in product(s_list, b_list, lambda_list):
            input_paras.append((data, s_tr, s_te, s, num_block, lambda_,
                                max_epochs, fold_i, sf_ii))
    pool = multiprocessing.Pool(processes=num_cpus)
    results_pool = pool.map(run_single_test, input_paras)
    pool.close()
    pool.join()

    sub_res = dict()
    for item in results_pool:
        s, num_blocks, lambda_, re, fold_i, subfold_i = item
        if subfold_i not in sub_res:
            sub_res[subfold_i] = []
        sub_res[subfold_i].append((s, num_blocks, lambda_, re))
    for sf_ii in sub_res:
        res = {_: dict() for _ in method_list}
        for _ in method_list:
            res[_]['s_list'] = s_list
            res[_]['b_list'] = b_list
            res[_]['lambda_list'] = lambda_list
            res[_]['auc'] = dict()
            res[_]['acc'] = dict()
            res[_]['bacc'] = dict()
            res[_]['perf'] = dict()
            res[_]['w_hat'] = {(s, num_blocks, lambda_): None
                               for (s, num_blocks, lambda_) in
                               product(s_list, b_list, lambda_list)}
        for s, num_blocks, lambda_, re in sub_res[sf_ii]:
            for _ in method_list:
                res[_]['auc'][(s, num_blocks, lambda_)] = re[_]['auc']
                res[_]['acc'][(s, num_blocks, lambda_)] = re[_]['acc']
                res[_]['bacc'][(s, num_blocks, lambda_)] = re[_]['bacc']
                res[_]['perf'][(s, num_blocks, lambda_)] = re[_]['perf']
                res[_]['w_hat'][(s, num_blocks, lambda_)] = re[_]['w_hat']
        for _ in method_list:
            for (s, num_blocks, lambda_) in \
                    product(s_list, b_list, lambda_list):
                key_para = (s, num_blocks, lambda_)
                s_auc[_][key_para] += res[_]['auc'][key_para]
                s_acc[_][key_para] += res[_]['acc'][key_para]
                s_bacc[_][key_para] += res[_]['bacc'][key_para]
    # tune by balanced accuracy
    s_star = dict()
    for _ in method_list:
        s_star[_] = min(s_bacc[_], key=s_bacc[_].get)
    return s_star, s_bacc


def run_parallel_te(
        data, tr_idx, te_idx, s_list, b_list, lambda_list, max_epochs,
        num_cpus):
    method_list = ['sto-iht', 'graph-sto-iht', 'iht', 'graph-iht']
    res = {_: dict() for _ in method_list}
    for _ in method_list:
        res[_]['s_list'] = s_list
        res[_]['b_list'] = b_list
        res[_]['lambda_list'] = lambda_list
        res[_]['auc'] = dict()
        res[_]['acc'] = dict()
        res[_]['bacc'] = dict()
        res[_]['perf'] = dict()
        res[_]['w_hat'] = {(s, num_blocks, lambda_): None
                           for (s, num_blocks, lambda_) in
                           product(s_list, b_list, lambda_list)}
    input_paras = [(data, tr_idx, te_idx, s, num_block, lambda_, max_epochs,
                    '', '') for s, num_block, lambda_ in
                   product(s_list, b_list, lambda_list)]
    pool = multiprocessing.Pool(processes=num_cpus)
    results_pool = pool.map(run_single_test, input_paras)
    pool.close()
    pool.join()
    for s, num_blocks, lambda_, re, fold_i, subfold_i in results_pool:
        for _ in method_list:
            res[_]['auc'][(s, num_blocks, lambda_)] = re[_]['auc']
            res[_]['acc'][(s, num_blocks, lambda_)] = re[_]['acc']
            res[_]['bacc'][(s, num_blocks, lambda_)] = re[_]['bacc']
            res[_]['perf'][(s, num_blocks, lambda_)] = re[_]['perf']
            res[_]['w_hat'][(s, num_blocks, lambda_)] = re[_]['w_hat']
    return res


def get_single_data(trial_i, root_input):
    import scipy.io as sio
    cancer_related_genes = {
        4288: 'MKI67', 1026: 'CDKN1A', 472: 'ATM', 7033: 'TFF3', 2203: 'FBP1',
        7494: 'XBP1', 1824: 'DSC2', 1001: 'CDH3', 11200: 'CHEK2',
        7153: 'TOP2A', 672: 'BRCA1', 675: 'BRCA2', 580: 'BARD1', 9: 'NAT1',
        771: 'CA12', 367: 'AR', 7084: 'TK2', 5892: 'RAD51D', 2625: 'GATA3',
        7155: 'TOP2B', 896: 'CCND3', 894: 'CCND2', 10551: 'AGR2',
        3169: 'FOXA1', 2296: 'FOXC1'}
    data = dict()
    f_name = 'overlap_data_%02d.mat' % trial_i
    re = sio.loadmat(root_input + f_name)['save_data'][0][0]
    data['data_X'] = np.asarray(re['data_X'], dtype=np.float64)
    data_y = [_[0] for _ in re['data_Y']]
    data['data_Y'] = np.asarray(data_y, dtype=np.float64)
    data_edges = [[_[0] - 1, _[1] - 1] for _ in re['data_edges']]
    data['data_edges'] = np.asarray(data_edges, dtype=int)
    data_pathways = [[_[0], _[1]] for _ in re['data_pathways']]
    data['data_pathways'] = np.asarray(data_pathways, dtype=int)
    data_entrez = [_[0] for _ in re['data_entrez']]
    data['data_entrez'] = np.asarray(data_entrez, dtype=int)
    data['data_splits'] = {i: dict() for i in range(5)}
    data['data_subsplits'] = {i: {j: dict() for j in range(5)}
                              for i in range(5)}
    for i in range(5):
        xx = re['data_splits'][0][i][0][0]['train']
        data['data_splits'][i]['train'] = [_ - 1 for _ in xx[0]]
        xx = re['data_splits'][0][i][0][0]['test']
        data['data_splits'][i]['test'] = [_ - 1 for _ in xx[0]]
        for j in range(5):
            xx = re['data_subsplits'][0][i][0][j]['train'][0][0]
            data['data_subsplits'][i][j]['train'] = [_ - 1 for _ in xx[0]]
            xx = re['data_subsplits'][0][i][0][j]['test'][0][0]
            data['data_subsplits'][i][j]['test'] = [_ - 1 for _ in xx[0]]
    re_path = [_[0] for _ in re['re_path_varInPath']]
    data['re_path_varInPath'] = np.asarray(re_path)
    re_path_entrez = [_[0] for _ in re['re_path_entrez']]
    data['re_path_entrez'] = np.asarray(re_path_entrez)
    re_path_ids = [_[0] for _ in re['re_path_ids']]
    data['re_path_ids'] = np.asarray(re_path_ids)
    re_path_lambdas = [_ for _ in re['re_path_lambdas'][0]]
    data['re_path_lambdas'] = np.asarray(re_path_lambdas)
    re_path_groups = [_[0][0] for _ in re['re_path_groups_lasso'][0]]
    data['re_path_groups_lasso'] = np.asarray(re_path_groups)
    re_path_groups_overlap = [_[0][0] for _ in re['re_path_groups_overlap'][0]]
    data['re_path_groups_overlap'] = np.asarray(re_path_groups_overlap)
    re_edge = [_[0] for _ in re['re_edge_varInGraph']]
    data['re_edge_varInGraph'] = np.asarray(re_edge)
    re_edge_entrez = [_[0] for _ in re['re_edge_entrez']]
    data['re_edge_entrez'] = np.asarray(re_edge_entrez)
    data['re_edge_groups_lasso'] = np.asarray(re['re_edge_groups_lasso'])
    data['re_edge_groups_overlap'] = np.asarray(re['re_edge_groups_overlap'])
    for method in ['re_path_re_lasso', 're_path_re_overlap',
                   're_edge_re_lasso', 're_edge_re_overlap']:
        res = {fold_i: dict() for fold_i in range(5)}
        for fold_ind, fold_i in enumerate(range(5)):
            res[fold_i]['lambdas'] = re[method][0][fold_i]['lambdas'][0][0][0]
            res[fold_i]['kidx'] = re[method][0][fold_i]['kidx'][0][0][0]
            res[fold_i]['kgroups'] = re[method][0][fold_i]['kgroups'][0][0][0]
            res[fold_i]['kgroupidx'] = re[method][0][fold_i]['kgroupidx'][0][0]
            res[fold_i]['groups'] = re[method][0][fold_i]['groups'][0]
            res[fold_i]['sbacc'] = re[method][0][fold_i]['sbacc'][0]
            res[fold_i]['AS'] = re[method][0][fold_i]['AS'][0]
            res[fold_i]['completeAS'] = re[method][0][fold_i]['completeAS'][0]
            res[fold_i]['lstar'] = re[method][0][fold_i]['lstar'][0][0][0][0]
            res[fold_i]['auc'] = re[method][0][fold_i]['auc'][0]
            res[fold_i]['acc'] = re[method][0][fold_i]['acc'][0]
            res[fold_i]['bacc'] = re[method][0][fold_i]['bacc'][0]
            res[fold_i]['perf'] = re[method][0][fold_i]['perf'][0][0]
            res[fold_i]['pred'] = re[method][0][fold_i]['pred']
            res[fold_i]['Ws'] = re[method][0][fold_i]['Ws'][0][0]
            res[fold_i]['oWs'] = re[method][0][fold_i]['oWs'][0][0]
            res[fold_i]['nextGrad'] = re[method][0][fold_i]['nextGrad'][0]
        data[method] = res
    import networkx as nx
    g = nx.Graph()
    ind_pathways = {_: i for i, _ in enumerate(data['data_entrez'])}
    all_nodes = {ind_pathways[_]: '' for _ in data['re_path_entrez']}
    maximum_nodes, maximum_list_edges = set(), []
    for edge in data['data_edges']:
        if edge[0] in all_nodes and edge[1] in all_nodes:
            g.add_edge(edge[0], edge[1])
    isolated_genes = set()
    maximum_genes = set()
    for cc in nx.connected_component_subgraphs(g):
        if len(cc) <= 5:
            for item in list(cc):
                isolated_genes.add(data['data_entrez'][item])
        else:
            for item in list(cc):
                maximum_nodes = set(list(cc))
                maximum_genes.add(data['data_entrez'][item])
    maximum_nodes = np.asarray(list(maximum_nodes))
    subgraph = nx.Graph()
    for edge in data['data_edges']:
        if edge[0] in maximum_nodes and edge[1] in maximum_nodes:
            if edge[0] != edge[1]:  # remove some self-loops
                maximum_list_edges.append(edge)
            subgraph.add_edge(edge[0], edge[1])
    data['map_entrez'] = np.asarray([data['data_entrez'][_]
                                     for _ in maximum_nodes])
    data['edges'] = np.asarray(maximum_list_edges, dtype=int)
    data['costs'] = np.asarray([1.] * len(maximum_list_edges),
                               dtype=np.float64)
    data['x'] = data['data_X'][:, maximum_nodes]
    data['y'] = data['data_Y']
    data['nodes'] = np.asarray(range(len(maximum_nodes)), dtype=int)
    data['cancer_related_genes'] = cancer_related_genes
    for edge_ind, edge in enumerate(data['edges']):
        uu = list(maximum_nodes).index(edge[0])
        vv = list(maximum_nodes).index(edge[1])
        data['edges'][edge_ind][0] = uu
        data['edges'][edge_ind][1] = vv
    method_list = ['re_path_re_lasso', 're_path_re_overlap',
                   're_edge_re_lasso', 're_edge_re_overlap']
    found_set = {method: set() for method in method_list}
    for method in method_list:
        for fold_i in range(5):
            best_lambda = data[method][fold_i]['lstar']
            kidx = data[method][fold_i]['kidx']
            re = list(data[method][fold_i]['lambdas']).index(best_lambda)
            ws = data[method][fold_i]['oWs'][:, re]
            for item in [kidx[_] for _ in np.nonzero(ws[1:])[0]]:
                if item in cancer_related_genes:
                    found_set[method].add(cancer_related_genes[item])
    data['found_related_genes'] = found_set
    return data


def run_test(folding_i, num_cpus, root_input, root_output):
    n_folds, max_epochs = 5, 10
    s_list = range(5, 100, 5)  # sparsity list
    b_list = [1, 2]  # number of block list.
    lambda_list = [1e-3, 1e-4]
    method_list = ['sto-iht', 'graph-sto-iht', 'iht', 'graph-iht']
    cv_res = {_: dict() for _ in range(n_folds)}
    for fold_i in range(n_folds):
        data = get_single_data(folding_i, root_input)
        tr_idx = data['data_splits'][fold_i]['train']
        te_idx = data['data_splits'][fold_i]['test']
        f_data = data.copy()
        tr_data = dict()
        tr_data['x'] = f_data['x'][tr_idx, :]
        tr_data['y'] = f_data['y'][tr_idx]
        tr_data['data_entrez'] = f_data['data_entrez']
        f_data['x'] = data['x']
        # data normalization
        x_mean = np.tile(np.mean(f_data['x'], axis=0), (len(f_data['x']), 1))
        x_std = np.tile(np.std(f_data['x'], axis=0), (len(f_data['x']), 1))
        f_data['x'] = np.nan_to_num(np.divide(f_data['x'] - x_mean, x_std))
        f_data['edges'] = data['edges']
        f_data['costs'] = data['costs']
        cv_res[fold_i]['s_list'] = s_list
        cv_res[fold_i]['b_list'] = b_list
        cv_res[fold_i]['lambda_list'] = lambda_list
        s_star, s_bacc = run_parallel_tr(
            f_data, s_list, b_list, lambda_list, max_epochs, num_cpus, fold_i)
        for _ in method_list:
            cv_res[fold_i][_] = dict()
            cv_res[fold_i][_]['s_bacc'] = s_bacc[_]
        res = run_parallel_te(
            f_data, tr_idx, te_idx, s_list, b_list, lambda_list,
            max_epochs, num_cpus)
        for _ in method_list:
            best_para = s_star[_]
            cv_res[fold_i][_]['s_star'] = best_para
            cv_res[fold_i][_]['auc'] = res[_]['auc'][best_para]
            cv_res[fold_i][_]['acc'] = res[_]['acc'][best_para]
            cv_res[fold_i][_]['bacc'] = res[_]['bacc'][best_para]
            cv_res[fold_i][_]['perf'] = res[_]['bacc'][best_para]
            cv_res[fold_i][_]['w_hat'] = res[_]['w_hat'][best_para]
            cv_res[fold_i][_]['map_entrez'] = data['map_entrez']
    f_name = 'results_exp_bc_%02d_%02d.pkl' % (folding_i, max_epochs)
    pickle.dump(cv_res, open(root_output + f_name, 'wb'))


def summarize_data(trial_list, num_iterations, root_output):
    sum_data = dict()
    cancer_related_genes = {
        4288: 'MKI67', 1026: 'CDKN1A', 472: 'ATM', 7033: 'TFF3', 2203: 'FBP1',
        7494: 'XBP1', 1824: 'DSC2', 1001: 'CDH3', 11200: 'CHEK2',
        7153: 'TOP2A', 672: 'BRCA1', 675: 'BRCA2', 580: 'BARD1', 9: 'NAT1',
        771: 'CA12', 367: 'AR', 7084: 'TK2', 5892: 'RAD51D', 2625: 'GATA3',
        7155: 'TOP2B', 896: 'CCND3', 894: 'CCND2', 10551: 'AGR2',
        3169: 'FOXA1', 2296: 'FOXC1'}
    for trial_i in trial_list:
        sum_data[trial_i] = dict()
        f_name = root_output + 'results_exp_bc_%02d_%03d.pkl' % \
                 (trial_i, num_iterations)
        data = pickle.load(open(f_name))
        for method in ['graph-sto-iht', 'sto-iht', 'graph-iht', 'iht']:
            sum_data[trial_i][method] = dict()
            auc, bacc, non_zeros_list, found_genes = [], [], [], []
            for fold_i in data:
                auc.append(data[fold_i][method]['auc'])
                bacc.append(data[fold_i][method]['bacc'])
                wt = data[fold_i][method]['w_hat']
                non_zeros_list.append(len(np.nonzero(wt[:len(wt) - 1])[0]))
                sum_data[trial_i][method]['w_hat_%d' % fold_i] = \
                    wt[:len(wt) - 1]
                for element in np.nonzero(wt[:len(wt) - 1])[0]:
                    found_genes.append(
                        data[fold_i][method]['map_entrez'][element])
            found_genes = [cancer_related_genes[_]
                           for _ in found_genes
                           if _ in cancer_related_genes]
            sum_data[trial_i][method]['auc'] = auc
            sum_data[trial_i][method]['bacc'] = bacc
            sum_data[trial_i][method]['num_nonzeros'] = non_zeros_list
            sum_data[trial_i][method]['found_genes'] = found_genes

    return sum_data


def show_test(trials_list, num_iterations, root_input, root_output,
              latex_flag=True):
    sum_data = summarize_data(trials_list, num_iterations, root_output)
    all_data = pickle.load(open(root_input + 'overlap_data_summarized.pkl'))
    for trial_i in sum_data:
        for method in ['graph-sto-iht', 'sto-iht', 'graph-iht', 'iht']:
            all_data[trial_i]['re_%s' % method] = sum_data[trial_i][method]
        for method in ['re_graph-sto-iht', 're_sto-iht',
                       're_graph-iht', 're_iht']:
            re = all_data[trial_i][method]['found_genes']
            all_data[trial_i]['found_related_genes'][method] = set(re)
    method_list = ['re_path_re_lasso', 're_path_re_overlap',
                   're_edge_re_lasso', 're_edge_re_overlap',
                   're_sto-iht', 're_graph-sto-iht', 're_iht', 're_graph-iht']
    all_involved_genes = {method: set() for method in method_list}
    for trial_i in sum_data:
        for method in ['graph-sto-iht', 'sto-iht', 'graph-iht', 'iht']:
            all_data[trial_i]['re_%s' % method] = sum_data[trial_i][method]
        for method in ['re_graph-sto-iht', 're_sto-iht', 're_graph-iht',
                       're_iht']:
            re = all_data[trial_i][method]['found_genes']
            all_data[trial_i]['found_related_genes'][method] = set(re)
        for method in ['re_path_re_lasso', 're_path_re_overlap',
                       're_edge_re_lasso', 're_edge_re_overlap']:
            for fold_i in range(5):
                re = np.nonzero(all_data[trial_i][method]['ws_%d' % fold_i])
                all_involved_genes[method] = set(re[0]).union(
                    all_involved_genes[method])
        for method in ['re_sto-iht', 're_graph-sto-iht']:
            for fold_i in range(5):
                re = np.nonzero(all_data[trial_i][method]['w_hat_%d' % fold_i])
                all_involved_genes[method] = set(re[0]).union(
                    all_involved_genes[method])
    for method in method_list:
        all_involved_genes[method] = len(all_involved_genes[method])
    print('_' * 122)
    print('_' * 122)
    for metric in ['bacc', 'auc', 'num_nonzeros']:
        mean_dict = {method: [] for method in method_list}
        print(' '.join(['-' * 54, '%12s' % metric, '-' * 54]))
        print('            Path-Lasso    Path-Overlap '),
        print('Edge-Lasso    Edge-Overlap  StoIHT        '
              'GraphStoIHT   IHT           GraphIHT')
        for folding_i in trials_list:
            each_re = all_data[folding_i]
            print('Folding_%02d ' % folding_i),
            for method in method_list:
                x1 = float(np.mean(each_re[method][metric]))
                x2 = float(np.std(each_re[method][metric]))
                mean_dict[method].extend(each_re[method][metric])
                if metric == 'num_nonzeros':
                    print('%05.1f,%05.2f  ' % (x1, x2)),
                else:
                    print('%.3f,%.3f  ' % (x1, x2)),
            print('')
        print('Averaged   '),
        for method in method_list:
            x1 = float(np.mean(mean_dict[method]))
            x2 = float(np.std(mean_dict[method]))
            if metric == 'num_nonzeros':
                print('%05.1f,%05.2f  ' % (x1, x2)),
            else:
                print('%.3f,%.3f  ' % (x1, x2)),
        print('')
    print('_' * 122)
    if latex_flag:  # print latex table.
        print('\n\n')
        print('_' * 164)
        print('_' * 164)
        print(' '.join(['-' * 75, 'latex tables', '-' * 75]))
        caption_list = [
            'Balanced Classification Error on the breast cancer dataset.',
            'AUC score on the breast cancer dataset.',
            'Number of nonzeros on the breast cancer dataset.']
        for index_metrix, metric in enumerate(['bacc', 'auc', 'num_nonzeros']):
            print(' '.join(['-' * 75, '%12s' % metric, '-' * 75]))
            print(
                    '\\begin{table*}[ht!]\n\\caption{%s}\n'
                    '\centering\n\scriptsize\n\\begin{tabular}{ccccccccccc}' %
                    caption_list[index_metrix])
            print('\hline')
            mean_dict = {method: [] for method in method_list}

            print(' & '.join(
                ['Fold Choice',
                 '$\ell_1$-\\textsc{Pathway}',
                 '$\ell^1/\ell^2$-\\textsc{Pathway}',
                 '$\ell_1$-\\textsc{Edge}',
                 '$\ell^1/\ell^2$-\\textsc{Edge}',
                 '\\textsc{StoIHT}',
                 '\\textsc{GraphStoIHT}',
                 '\\textsc{IHT}',
                 '\\textsc{GraphIHT}'])),
            print('\\\\')
            print('\hline')
            for folding_i in trials_list:
                row_list = []
                find_min = [np.inf]
                find_max = [-np.inf]
                each_re = all_data[folding_i]
                row_list.append('Folding %02d' % folding_i)
                for method in method_list:
                    x1 = float(np.mean(each_re[method][metric]))
                    find_min.append(x1)
                    find_max.append(x1)
                    x2 = float(np.std(each_re[method][metric]))
                    mean_dict[method].extend(each_re[method][metric])
                    if metric == 'num_nonzeros':
                        row_list.append('%05.1f$\pm$%05.2f' % (x1, x2))
                    else:
                        row_list.append('%.3f$\pm$%.3f' % (x1, x2))
                if metric == 'bacc':
                    min_index = np.argmin(find_min)
                    original_string = row_list[int(min_index)]
                    mean_std = original_string.split('$\pm$')
                    row_list[int(min_index)] = '\\textbf{%s}$\pm$%s' % (
                        mean_std[0], mean_std[1])
                elif metric == 'auc':
                    max_index = np.argmax(find_max)
                    original_string = row_list[int(max_index)]
                    mean_std = original_string.split('$\pm$')
                    row_list[int(max_index)] = '\\textbf{%s}$\pm$%s' % (
                        mean_std[0], mean_std[1])
                print(' & '.join(row_list)),
                print('\\\\')
            print('\hline')
            row_list = ['Averaged ']
            find_min = [np.inf]
            find_max = [-np.inf]
            for method in method_list:
                x1 = float(np.mean(mean_dict[method]))
                x2 = float(np.std(mean_dict[method]))
                find_min.append(x1)
                find_max.append(x1)
                if metric == 'num_nonzeros':
                    row_list.append('%05.1f$\pm$%05.2f' % (x1, x2))
                else:
                    row_list.append('%.3f$\pm$%.3f' % (x1, x2))
            if metric == 'bacc':
                min_index = np.argmin(find_min)
                original_string = row_list[int(min_index)]
                mean_std = original_string.split('$\pm$')
                row_list[int(min_index)] = '\\textbf{%s}$\pm$%s' % (
                    mean_std[0], mean_std[1])
            elif metric == 'auc':
                max_index = np.argmax(find_max)
                original_string = row_list[int(max_index)]
                mean_std = original_string.split('$\pm$')
                row_list[int(max_index)] = '\\textbf{%s}$\pm$%s' % (
                    mean_std[0], mean_std[1])
            print(' & '.join(row_list)),
            print('\\\\')
            print('\hline')
            print('\end{tabular}\n\end{table*}')
        print('_' * 164)
    found_genes = {method: set() for method in method_list}
    for folding_i in trials_list:
        for method in method_list:
            re = all_data[folding_i]['found_related_genes'][method]
            found_genes[method] = set(re).union(found_genes[method])
    print('\n\n')
    print('_' * 85)
    print('_' * 85)
    print(' '.join(['-' * 35, 'related genes', '-' * 35]))
    for method in method_list:
        print('%-20s: %-s' % (method, ' '.join(list(found_genes[method]))))
    print('_' * 85)


def main():
    command = sys.argv[1]
    if command == 'run_test':
        num_cpus = int(sys.argv[2])
        trial_start = int(sys.argv[3])
        trial_end = int(sys.argv[4])
        for folding_i in range(trial_start, trial_end):
            run_test(folding_i=folding_i, num_cpus=num_cpus,
                     root_input='data/', root_output='results/')
    elif command == 'show_test':
        folding_list = range(20)
        num_iterations = 50
        show_test(folding_list, num_iterations,
                  root_input='data/', root_output='results/')


if __name__ == "__main__":
    main()
