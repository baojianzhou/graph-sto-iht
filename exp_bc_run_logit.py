# -*- coding: utf-8 -*-

import os
import csv
import sys
import time
import math
import pickle
import numpy as np
import multiprocessing
from os import path
from itertools import product
from numpy.random import randint

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from algo_wrapper.algo_wrapper import algo_head_tail_binsearch
from algo_wrapper.base import logistic_predict
from algo_wrapper.base import logit_loss_bl
from algo_wrapper.base import logit_loss_grad_bl


def algo_graph_sto_iht_backtracking(
        x_tr, y_tr, w0, max_epochs, h_low, h_high, t_low, t_high, edges,
        costs, root, g, max_num_iter, verbose, num_blocks, lambda_):
    np.random.seed()  # do not forget it.
    w_hat = np.copy(w0)
    (m, p) = x_tr.shape
    # if the block size is too large. just use single block
    b = int(m) / int(num_blocks)
    num_epochs = 0
    np_ = np.sum(y_tr == 1)
    nn_ = np.sum(y_tr == -1)
    cp = float(nn_) / float(len(y_tr))
    cn = float(np_) / float(len(y_tr))
    w_error_list, loss_list = [], []
    for epoch_i in range(max_epochs):
        num_epochs += 1
        for ind, _ in enumerate(range(num_blocks)):
            ii = randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            x_tr_b, y_tr_b = x_tr[block, :], y_tr[block]
            loss_sto, grad_sto = logit_loss_grad_bl(
                x_tr=x_tr_b, y_tr=y_tr_b, wt=w_hat,
                l2_reg=lambda_, cp=cp, cn=cn)
            h_nodes, p_grad = algo_head_tail_binsearch(
                edges, grad_sto[:p], costs, g, root, h_low, h_high,
                max_num_iter, verbose)
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
            t_nodes, proj_bt = algo_head_tail_binsearch(
                edges, bt_sto[:p], costs, g, root, t_low, t_high, max_num_iter,
                verbose)
            w_hat[:p] = proj_bt[:p]
            w_hat[p] = w_hat[p] - ad_step * grad_sto[p]  # intercept.
            print('iter: %03d sparsity: %02d num_blocks: %02d loss_sto: %.4f '
                  'head_nodes: %03d tail_nodes: %03d' %
                  (epoch_i * num_blocks + ind, t_low, num_blocks, loss_sto,
                   len(h_nodes), len(t_nodes)))
            loss_list.append(loss_sto)
            w_error_list.append(np.linalg.norm(w_hat))

        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        # we reach a local minimum point.
        if len(np.unique(loss_list[-5:])) == 1 and \
                len(np.unique(w_error_list[-5:])) == 1 and epoch_i >= 10:
            print('found to be at a local minimal!!')
            break
    return w_hat


def algo_sto_iht_backtracking(
        x_tr, y_tr, w0, max_epochs, s, num_blocks, lambda_, verbose=0):
    np.random.seed()  # do not forget it.
    w_hat = w0
    (m, p) = x_tr.shape
    b = int(m) / int(num_blocks)
    num_epochs = 0
    np_ = np.sum(y_tr == 1)
    nn_ = np.sum(y_tr == -1)
    cp = float(nn_) / float(len(y_tr))
    cn = float(np_) / float(len(y_tr))
    w_error_list, loss_list = [], []
    for epoch_i in range(max_epochs):
        num_epochs += 1
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
            loss_list.append(loss_sto)
            if verbose > 0:
                print('iter: %03d sparsity: %02d loss_sto: %.10f'
                      % (epoch_i * num_blocks + ind, s, loss_sto))
            w_error_list.append(np.linalg.norm(w_hat))

        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        # we reach a local minimum point.
        if len(np.unique(loss_list[-5:])) == 1 and \
                len(np.unique(w_error_list[-5:])) == 1 and epoch_i >= 10:
            print('found to be at a local minimal!!')
            break
    return w_hat


def algo_sto_iht(
        x_tr, y_tr, w0, lr, max_epochs, s, num_blocks, with_replace=True,
        verbose=0):
    np.random.seed()  # do not forget it.
    w_hat = np.copy(w0)
    (m, p) = x_tr.shape
    # if the block size is too large. just use single block
    b = int(m) / int(num_blocks)
    num_epochs = 0
    np_ = np.sum(y_tr == 1)
    nn_ = np.sum(y_tr == -1)
    cp = float(nn_) / float(len(y_tr))
    cn = float(np_) / float(len(y_tr))
    w_error_list, loss_list = [], []
    for epoch_i in range(max_epochs):
        rand_perm = np.random.permutation(num_blocks)
        for ind, _ in enumerate(range(num_blocks)):
            ii = randint(0, num_blocks) if with_replace else rand_perm[_]
            block = range(b * ii, b * (ii + 1))
            x_tr_b, y_tr_b = x_tr[block, :], y_tr[block]
            loss_sto, grad_sto = logit_loss_grad_bl(
                x_tr=x_tr_b, y_tr=y_tr_b, wt=w_hat, l2_reg=1e-4, cp=cp, cn=cn)
            bt_sto = w_hat - lr * grad_sto
            bt_sto[np.argsort(np.abs(bt_sto))[:p - s]] = 0.
            w_hat = bt_sto
            if verbose > 0:
                print('iter: %03d sparsity: %02d loss_sto: %.10f'
                      % (epoch_i * num_blocks + ind, s, loss_sto))
            w_error_list.append(np.linalg.norm(w_hat))
        num_epochs += 1

        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        # we reach a local minimum point.
        if len(np.unique(loss_list[-5:])) == 1 and \
                len(np.unique(w_error_list[-5:])) == 1 and epoch_i >= 10:
            print('found to be at a local minimal!!')
            break
    return w_hat


def algo_graph_sto_iht(
        x_tr, y_tr, w0, lr, max_epochs, h_low, h_high, t_low, t_high, edges,
        costs, root, g, max_num_iter, verbose, num_blocks, with_replace=True):
    np.random.seed()  # do not forget it.
    w_hat = np.copy(w0)
    (m, p) = x_tr.shape
    # if the block size is too large. just use single block
    b = int(m) / int(num_blocks)
    num_epochs = 0
    np_ = np.sum(y_tr == 1)
    nn_ = np.sum(y_tr == -1)
    cp = float(nn_) / float(len(y_tr))
    cn = float(np_) / float(len(y_tr))
    for epoch_i in range(max_epochs):
        rand_perm = np.random.permutation(num_blocks)
        for ind, _ in enumerate(range(num_blocks)):
            ii = randint(0, num_blocks) if with_replace else rand_perm[_]
            block = range(b * ii, b * (ii + 1))
            x_tr_b, y_tr_b = x_tr[block, :], y_tr[block]
            loss_sto, grad_sto = logit_loss_grad_bl(
                x_tr=x_tr_b, y_tr=y_tr_b, wt=w_hat, l2_reg=1e-4, cp=cp, cn=cn)
            h_nodes, p_grad = algo_head_tail_binsearch(
                edges, grad_sto[:p], costs, g, root, h_low, h_high,
                max_num_iter, verbose)
            bt_sto = np.zeros_like(w_hat)
            bt_sto[:p] = w_hat[:p] - lr * p_grad[:p]
            t_nodes, proj_bt = algo_head_tail_binsearch(
                edges, bt_sto[:p], costs, g, root, t_low, t_high, max_num_iter,
                verbose)
            w_hat[:p] = proj_bt[:p]
            w_hat[p] = w_hat[p] - lr * grad_sto[p]  # intercept.
            print('iter: %03d sparsity: %02d num_blocks: %02d loss_sto: %.4f '
                  'head_nodes: %03d tail_nodes: %03d' %
                  (epoch_i * num_blocks + ind, t_low, num_blocks, loss_sto,
                   len(h_nodes), len(t_nodes)))
        num_epochs += 1
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


def get_data(nfolds):
    data = dict()
    splits = dict()
    raw_data = pickle.load(open(root_input + 'data.pkl'))
    x = raw_data['x']
    y = raw_data['y']
    entrez = raw_data['entrez']
    data['x'] = np.asarray(x, dtype=float)
    data['y'] = np.asarray(y, dtype=float)
    data['entrez'] = np.asarray([_[0] for _ in entrez], dtype=int)

    nodes, edges = set(), []
    with open(root_p + 'raw/edge.txt') as csvfile:
        edge_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in edge_reader:
            uu, vv = int(row[0]) - 1, int(row[1]) - 1
            nodes.add(uu)
            nodes.add(vv)
            edges.append([uu, vv])
    edges = np.asarray(edges, dtype=int)
    print('number of nodes: %d' % len(nodes))
    print(len(data['x'][0]))
    print(np.max(edges))
    var_in_graph = np.unique(edges.flatten())
    print('nodes in x: %d' % len(var_in_graph))
    data['edges'] = edges

    sub_splits = dict()
    super_splits = cv_split_bal(y, nfolds[0])
    for i in range(0, nfolds[0]):
        sub_splits[i] = cv_split_bal(y[super_splits[i]['x_tr']], nfolds[1])

    splits['splits'] = super_splits
    splits['sub_splits'] = sub_splits
    print(np.shape(data['x']))
    print(np.shape(data['y']))
    print(np.shape(data['entrez']))
    return data, splits


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
    data, tr_idx, te_idx, ii, s, jj, num_blocks, kk, lambda_, num_iterations = para
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
    # ----------------------------------------------------------------
    w_hat = algo_sto_iht_backtracking(
        x_tr, y_tr, w0, num_iterations, s, 1, lambda_, verbose=0)
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
    # ----------------------------------------------------------------
    w_hat = algo_sto_iht_backtracking(
        x_tr, y_tr, w0, num_iterations, s, num_blocks, lambda_, verbose=0)
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

    # ----------------------------------------------------------------
    tr_data = dict()
    tr_data['x'] = data['x'][tr_idx, :]
    tr_data['y'] = data['y'][tr_idx]
    te_data = dict()
    te_data['x'] = data['x'][te_idx, :]
    te_data['y'] = data['y'][te_idx]
    gamma = 0.1
    h_low, h_high = int(2 * s), int(2 * s * (1. + gamma))
    t_low, t_high = s, int(s * (1. + gamma))
    x_tr, y_tr = tr_data['x'], tr_data['y']
    w0 = np.zeros(np.shape(x_tr)[1] + 1)
    w_hat = algo_graph_sto_iht_backtracking(
        x_tr, y_tr, w0, num_iterations, h_low, h_high, t_low, t_high,
        data['edges'], data['costs'], -1, 1, 50, 0, num_blocks, lambda_)
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

    # ----------------------------------------------------------
    w_hat = algo_graph_sto_iht_backtracking(
        x_tr, y_tr, w0, 1, h_low, h_high, t_low, t_high,
        data['edges'], data['costs'], -1, 1, 50, 0, num_blocks, lambda_)
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
    return ii, s, jj, num_blocks, res


def run_parallel(
        data, tr_idx, te_idx, s_list, b_list, lambda_list, num_iters,
        num_cpus):
    method_list = ['sto-iht', 'graph-sto-iht', 'iht', 'graph-iht']
    res = {_: dict() for _ in method_list}
    for _ in method_list:
        res[_]['s_list'] = s_list
        res[_]['b_list'] = b_list
        res[_]['lambda_list'] = lambda_list
        res[_]['auc'] = np.zeros((len(s_list), len(b_list)))
        res[_]['acc'] = np.zeros((len(s_list), len(b_list)))
        res[_]['bacc'] = np.zeros((len(s_list), len(b_list)))
        res[_]['perf'] = np.zeros((len(s_list), len(b_list)))
        res[_]['w_hat'] = {(s, b): None for (s, b) in product(s_list, b_list)}
    input_paras = [(data, tr_idx, te_idx, ii, s, jj, num_block, kk, lambda_,
                    num_iters)
                   for (ii, s), (jj, num_block), (kk, lambda_) in
                   product(enumerate(s_list), enumerate(b_list),
                           enumerate(lambda_list))]
    pool = multiprocessing.Pool(processes=num_cpus)
    # data, tr_idx, te_idx, ii, s, jj, num_blocks, lambda_, num_iterations
    results_pool = pool.map(run_single_test, input_paras)
    pool.close()
    pool.join()
    for ii, s, jj, num_blocks, re in results_pool:
        for _ in method_list:
            res[_]['auc'][ii][jj] = re[_]['auc']
            res[_]['acc'][ii][jj] = re[_]['acc']
            res[_]['bacc'][ii][jj] = re[_]['bacc']
            res[_]['perf'][ii][jj] = re[_]['perf']
            res[_]['w_hat'][(s, num_blocks)] = re[_]['w_hat']
    return res


def load_dataset():
    method_list = ['re_graph_lasso', 're_graph_overlap', 're_path_lasso',
                   're_path_overlap']
    num_folding = 9
    all_data = {i: pickle.load(open(root_input + '')) for i in
                range(num_folding)}
    print('-' * 80)
    summary_results = {method: [] for method in method_list}
    for folding_i in range(num_folding):
        re = {method: [0.0, 0.0] for method in method_list}
        for method in method_list:
            perf_list = [all_data[folding_i][method][fold_i]['perf']
                         for fold_i in range(5)]
            summary_results[method].extend(perf_list)
            perf_list = np.asarray(perf_list)
            re[method][0] = np.mean(perf_list)
            re[method][1] = np.std(perf_list)
        print('Error folding_%02d   %.4f,%.4f   %.4f,%.4f   '
              '%.4f,%.4f   %.4f,%.4f' % (
                  folding_i + 1, re['re_graph_lasso'][0],
                  re['re_graph_lasso'][1], re['re_graph_overlap'][0],
                  re['re_graph_overlap'][1], re['re_path_lasso'][0],
                  re['re_path_lasso'][1], re['re_path_overlap'][0],
                  re['re_path_overlap'][1]))
    print('Summarized         %.4f,%.4f   %.4f,%.4f   '
          '%.4f,%.4f   %.4f,%.4f' % (
              float(np.mean(summary_results['re_graph_lasso'])),
              float(np.std(summary_results['re_graph_lasso'])),
              float(np.mean(summary_results['re_graph_overlap'])),
              float(np.std(summary_results['re_graph_overlap'])),
              float(np.mean(summary_results['re_path_lasso'])),
              float(np.std(summary_results['re_path_lasso'])),
              float(np.mean(summary_results['re_path_overlap'])),
              float(np.std(summary_results['re_path_overlap']))))


def get_single_graph_data(folding_i):
    f_name = root_output + 'results_bc_%02d_020.pkl' % folding_i
    results = pickle.load(open(f_name))
    return results


def run_test(trial_i, num_cpus, root_input, root_output):
    n_folds, num_iters = 5, 50
    s_list = range(5, 100, 5)  # sparsity list
    b_list = [1, 2]  # number of block list.
    lambda_list = [1e-4]
    method_list = ['sto-iht', 'graph-sto-iht', 'iht', 'graph-iht']
    cv_res = {_: dict() for _ in range(n_folds)}
    for fold_i in range(n_folds):
        f_name = root_input + 'overlap_data_%02d.pkl' % trial_i
        data = pickle.load(open(f_name))
        tr_idx = data['data_splits'][fold_i]['train']
        te_idx = data['data_splits'][fold_i]['test']
        f_data = data.copy()
        tr_data = dict()
        tr_data['x'] = f_data['x'][tr_idx, :]
        tr_data['y'] = f_data['y'][tr_idx]
        tr_data['data_entrez'] = f_data['data_entrez']
        f_data['x'] = data['x']
        x_mean = np.tile(np.mean(f_data['x'], axis=0), (len(f_data['x']), 1))
        x_std = np.tile(np.std(f_data['x'], axis=0), (len(f_data['x']), 1))
        f_data['x'] = np.nan_to_num(np.divide(f_data['x'] - x_mean, x_std))
        f_data['edges'] = data['edges']
        f_data['costs'] = data['costs']
        cv_res[fold_i]['s_list'] = s_list
        cv_res[fold_i]['b_list'] = b_list

        s_auc = {_: np.zeros((len(s_list), len(b_list))) for _ in method_list}
        s_acc = {_: np.zeros((len(s_list), len(b_list))) for _ in method_list}
        s_bacc = {_: np.zeros((len(s_list), len(b_list))) for _ in method_list}
        s_star = {_: None for _ in method_list}  # save the best.
        sub_res = dict()
        for sf_ii in range(len(data['data_subsplits'][fold_i])):
            s_tr = data['data_subsplits'][fold_i][sf_ii]['train']
            s_te = data['data_subsplits'][fold_i][sf_ii]['test']
            sub_res[sf_ii] = run_parallel(
                f_data, s_tr, s_te, s_list, b_list, lambda_list,
                num_iters, num_cpus)
            for _ in method_list:
                s_auc[_] += sub_res[sf_ii][_]['auc'] / 1. * n_folds
                s_acc[_] += sub_res[sf_ii][_]['acc'] / 1. * n_folds
                s_bacc[_] += sub_res[sf_ii][_]['bacc'] / 1. * n_folds
        for _ in method_list:
            s_star[_] = np.unravel_index(s_bacc[_].argmin(), s_bacc[_].shape)
            cv_res[fold_i][_] = dict()
            cv_res[fold_i][_]['s_bacc'] = s_bacc[_]
        res = run_parallel(
            f_data, tr_idx, te_idx, s_list, b_list, lambda_list,
            num_iters, num_cpus)
        for _ in method_list:
            best_s = s_star[_]
            cv_res[fold_i][_]['s_star'] = best_s
            cv_res[fold_i][_]['auc'] = res[_]['auc'][best_s]
            cv_res[fold_i][_]['acc'] = res[_]['acc'][best_s]
            cv_res[fold_i][_]['bacc'] = res[_]['bacc'][best_s]
            cv_res[fold_i][_]['perf'] = res[_]['bacc'][best_s]
            s, b = (s_list[best_s[0]], b_list[best_s[1]])
            cv_res[fold_i][_]['w_hat'] = res[_]['w_hat'][(s, b)]
            cv_res[fold_i][_]['map_entrez'] = data['map_entrez']
    f_name = 'results_bc_%02d_%03d.pkl' % (trial_i, num_iters)
    pickle.dump(cv_res, open(root_output + f_name, 'wb'))


def summarize_data(trial_list, num_iterations):
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
        f_name = root_output + 'results_bc_%02d_%03d.pkl' % \
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


def show_test(trials_list, num_iterations):
    sum_data1 = summarize_data(trials_list, num_iterations)
    all_data = pickle.load(open(root_input + 'summarized_data.pkl'))
    for trial_i in sum_data1:
        for method in ['graph-sto-iht', 'sto-iht', 'graph-iht', 'iht']:
            all_data[trial_i]['re_%s' % method] = sum_data1[trial_i][method]
        for method in ['re_graph-sto-iht', 're_sto-iht',
                       're_graph-iht', 're_iht']:
            re = all_data[trial_i][method]['found_genes']
            all_data[trial_i]['found_related_genes'][method] = set(re)
    method_list = ['re_path_re_lasso', 're_path_re_overlap',
                   're_edge_re_lasso', 're_edge_re_overlap',
                   're_sto-iht', 're_graph-sto-iht', 're_iht', 're_graph-iht']
    for metric in ['bacc', 'auc', 'num_nonzeros']:
        mean_dict = {method: [] for method in method_list}
        print('-' * 30 + metric + '-' * 30)
        print('            Path-Lasso   Path-Overlap'),
        print('Edge-Lasso   Edge-Overlap    StoIHT '
              'GraphStoIHT     IHT     GraphIHT')
        for folding_i in trials_list:
            each_re = all_data[folding_i]
            print('Folding_%02d ' % folding_i),
            for method in method_list:
                x1 = float(np.mean(each_re[method][metric]))
                x2 = float(np.std(each_re[method][metric]))
                mean_dict[method].extend(each_re[method][metric])
                if metric == 'num_nonzeros':
                    print('%05.1f,%05.2f ' % (x1, x2)),
                else:
                    print('%.3f,%.3f ' % (x1, x2)),
            print('')
        print('Averaged   '),
        for method in method_list:
            x1 = float(np.mean(mean_dict[method]))
            x2 = float(np.std(mean_dict[method]))
            if metric == 'num_nonzeros':
                print('%05.1f,%05.2f ' % (x1, x2)),
            else:
                print('%.3f,%.3f ' % (x1, x2)),
        print('')
    print('-' * 60)
    found_genes = {method: set() for method in method_list}
    for folding_i in trials_list:
        for method in method_list:
            re = all_data[folding_i]['found_related_genes'][method]
            found_genes[method] = set(re).union(found_genes[method])
    for method in method_list:
        print(method, list(found_genes[method]))


def show_test02(trials_list, num_iterations):
    sum_data1 = summarize_data(trials_list, num_iterations)
    all_data = pickle.load(open(root_input + 'summarized_data.pkl'))
    method_list = ['re_path_re_lasso', 're_path_re_overlap',
                   're_edge_re_lasso', 're_edge_re_overlap',
                   're_sto-iht', 're_graph-sto-iht']
    all_involved_genes = {method: set() for method in method_list}
    for trial_i in sum_data1:
        for method in ['graph-sto-iht', 'sto-iht', 'graph-iht', 'iht']:
            all_data[trial_i]['re_%s' % method] = sum_data1[trial_i][method]
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
    method_list = ['re_path_re_lasso', 're_path_re_overlap',
                   're_edge_re_lasso', 're_edge_re_overlap',
                   're_sto-iht', 're_graph-sto-iht', 're_iht', 're_graph-iht']
    for metric in ['bacc', 'auc', 'num_nonzeros']:
        mean_dict = {method: [] for method in method_list}
        print('-' * 30 + metric + '-' * 30)
        print('            Path-Lasso   Path-Overlap'),
        print('Edge-Lasso   Edge-Overlap    StoIHT '
              'GraphStoIHT     IHT     GraphIHT')
        for folding_i in trials_list:
            each_re = all_data[folding_i]
            print('Error Folding %02d &' % folding_i),
            for method in method_list:
                x1 = float(np.mean(each_re[method][metric]))
                x2 = float(np.std(each_re[method][metric]))
                mean_dict[method].extend(each_re[method][metric])
                if metric == 'num_nonzeros':
                    print('%05.1f$\pm$%05.2f &' % (x1, x2)),
                else:
                    print('%.3f$\pm$%.3f &' % (x1, x2)),
            print('\\\\')
        print('Averaged   &'),
        for method in method_list:
            x1 = float(np.mean(mean_dict[method]))
            x2 = float(np.std(mean_dict[method]))
            if metric == 'num_nonzeros':
                print('%05.1f$\pm$%05.2f &' % (x1, x2)),
            else:
                print('%.3f$\pm$%.3f &' % (x1, x2)),
        print('\\\\')
    print('-' * 60)
    found_genes = {method: set() for method in method_list}
    for folding_i in trials_list:
        for method in method_list:
            re = all_data[folding_i]['found_related_genes'][method]
            found_genes[method] = set(re).union(found_genes[method])
    for method in method_list:
        print(method, list(found_genes[method]))


def main():
    command = sys.argv[1]
    if command == 'run_test':
        trial_start = int(sys.argv[2])
        trial_end = int(sys.argv[3])
        for trial_i in range(trial_start, trial_end):
            num_cpus = int(sys.argv[4])
            run_test(trial_i=trial_i, num_cpus=num_cpus,
                     root_input='data/', root_output='results/')
    elif command == 'show_test':
        trials_list = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17]
        trials_list = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
        trials_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       16, 17]
        trials_list = range(0, 20)
        # trials_list = [0, 4, 8, 12, 16]
        num_iterations = 50
        show_test(trials_list, num_iterations)
    elif command == 'show_test02':
        trials_list = range(0, 20)
        # trials_list = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
        num_iterations = 50
        show_test02(trials_list, num_iterations)


if __name__ == "__main__":
    main()
