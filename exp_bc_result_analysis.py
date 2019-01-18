# -*- coding: utf-8 -*-
import csv
import os
import sys
import time
import pickle
import sys
import os
import math
from itertools import product
import numpy as np
from os import path
import multiprocessing

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from algo_wrapper.algo_wrapper import algo_head_tail_binsearch

root_p = '/network/rit/lab/ceashpc/bz383376/data/icml19/breast_cancer/'
if not os.path.exists(root_p):
    os.mkdir(root_p)
root_input = root_p + 'input/'
if not os.path.exists(root_input):
    os.mkdir(root_input)
root_output = root_p + 'output/'
if not os.path.exists(root_output):
    os.mkdir(root_output)
root_figs = root_p + 'figs/'
if not os.path.exists(root_figs):
    os.mkdir(root_figs)


def expit(x):
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


def logistic_predict(x_va, wt):
    """ To predict the probability for sample xi. {+1,-1} """
    pre_prob, y_pred, p = [], [], x_va.shape[1]
    for i in range(len(x_va)):
        pred_posi = expit(np.dot(wt[:p], x_va[i]) + wt[p])
        pred_nega = 1. - pred_posi
        if pred_posi >= pred_nega:
            y_pred.append(1)
        else:
            y_pred.append(-1)
        pre_prob.append(pred_posi)
    return np.asarray(pre_prob), np.asarray(y_pred)


def _log_logistic(x):
    """ return log( 1/(1+exp(x)) )"""
    out = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            out[i] = -np.log(1 + np.exp(-x[i]))
        else:
            out[i] = x[i] - np.log(1 + np.exp(x[i]))
    return out


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


def logit_loss_grad(x_tr, y_tr, wt, eta, cp, cn):
    """ return {+1,-1} Logistic (val,grad) on training samples. """
    assert len(wt) == (x_tr.shape[1] + 1)
    c, n, p = wt[-1], x_tr.shape[0], x_tr.shape[1]
    posi_idx = np.where(y_tr == 1)
    nega_idx = np.where(y_tr == -1)
    wt = wt[:p]
    yz = y_tr * (np.dot(x_tr, wt) + c)
    z = _expit(yz)
    loss = -np.sum(_log_logistic(yz)) + .5 * eta * np.dot(wt, wt)
    grad = np.zeros(p + 1)
    bl_y_tr = np.zeros_like(y_tr)
    bl_y_tr[posi_idx] = cp * np.asarray(y_tr[posi_idx], dtype=float)
    bl_y_tr[nega_idx] = cn * np.asarray(y_tr[nega_idx], dtype=float)
    # z0 = (z - 1) * y_tr
    z0 = (z - 1) * bl_y_tr
    grad[:p] = np.dot(x_tr.T, z0) + eta * wt
    grad[-1] = z0.sum()
    return loss / float(n), grad / float(n)


def algo_graph_sto_iht(
        x_tr, y_tr, w0, lr, max_epochs, h_low, h_high,
        t_low, t_high, edges, costs, root, g,
        max_num_iter, verbose, num_blocks, with_replace=True):
    np.random.seed()  # do not forget it.
    w_sto = np.copy(w0)
    w_batch = np.copy(w0)
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
        loss_batch, grad_batch = logit_loss_grad(
            x_tr=x_tr, y_tr=y_tr, wt=w_batch, eta=1e-4, cp=cp, cn=cn)
        batch_head_nodes, proj_grad_batch = algo_head_tail_binsearch(
            edges, grad_batch[:p], costs, g, root, h_low, h_high,
            max_num_iter, verbose)
        bt_batch = np.zeros_like(w_batch)
        bt_batch[:p] = w_batch[:p] - lr * proj_grad_batch
        batch_tail_nodes, proj_bt_batch = algo_head_tail_binsearch(
            edges, bt_batch[:p], costs, g, root, t_low, t_high, max_num_iter,
            verbose)
        w_batch[:p] = proj_bt_batch[:p]
        w_batch[p] = w_batch[p] - lr * grad_batch[p]
        for ind, ii in enumerate(range(num_blocks)):
            if not with_replace:
                ii = rand_perm[ii]
            else:
                ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            x_tr_b, y_tr_b = x_tr[block, :], y_tr[block]
            loss_sto, grad_sto = logit_loss_grad(
                x_tr=x_tr_b, y_tr=y_tr_b, wt=w_sto, eta=1e-4, cp=cp, cn=cn)
            sto_head_nodes, proj_sto_grad = algo_head_tail_binsearch(
                edges, grad_sto[:p], costs, g, root, h_low, h_high,
                max_num_iter, verbose)
            bt_sto = np.zeros_like(w_sto)
            bt_sto[:p] = w_sto[:p] - lr * proj_sto_grad[:p]
            sto_tail_nodes, proj_bt_sto = algo_head_tail_binsearch(
                edges, bt_sto[:p], costs, g, root, t_low, t_high, max_num_iter,
                verbose)
            w_sto[:p] = proj_bt_sto[:p]
            w_sto[p] = w_sto[p] - lr * grad_sto[p]
            print('iter: %03d sparsity: %02d loss_batch: %.4f loss_sto: '
                  '%.4f head_nodes: (%03d,%03d) tail_nodes: (%03d, %03d)' %
                  (epoch_i * num_blocks + ind, t_low, loss_batch, loss_sto,
                   len(batch_head_nodes), len(sto_head_nodes),
                   len(batch_tail_nodes), len(sto_tail_nodes)))
        num_epochs += 1
    return w_sto, w_batch


def algo_sto_iht(
        x_tr, y_tr, w0, lr, max_epochs, s, num_blocks, with_replace=True):
    np.random.seed()  # do not forget it.
    w_sto = np.copy(w0)
    w_batch = np.copy(w0)
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
        loss_batch, grad_batch = logit_loss_grad(
            x_tr=x_tr, y_tr=y_tr, wt=w_batch, eta=1e-4, cp=cp, cn=cn)
        bt_batch = w_batch - lr * grad_batch
        bt_batch[np.argsort(np.abs(grad_batch))[0:p - s]] = 0.
        w_batch = bt_batch
        for ind, ii in enumerate(range(num_blocks)):
            if not with_replace:
                ii = rand_perm[ii]
            else:
                ii = np.random.randint(0, num_blocks)
            block = range(b * ii, b * (ii + 1))
            x_tr_b, y_tr_b = x_tr[block, :], y_tr[block]
            loss_sto, grad_sto = logit_loss_grad(
                x_tr=x_tr_b, y_tr=y_tr_b, wt=w_sto, eta=1e-4, cp=cp, cn=cn)
            bt_sto = w_sto - lr * grad_sto
            bt_sto[np.argsort(np.abs(bt_sto))[0:p - s]] = 0.
            w_sto = bt_sto
            print('iter: %03d sparsity: %02d loss_batch: %.4f loss_sto: %.4f'
                  % (epoch_i * num_blocks + ind, s, loss_batch, loss_sto))
        num_epochs += 1
    return w_sto, w_batch


def print_re(best_auc_re, best_node_fm, pathway_id, sub_graph):
    print('-' * 80)
    print('pathway_id: %s, number of genes: %d' % (pathway_id, len(sub_graph)))
    print('auc: %.4f -- %.4f' % (best_auc_re['auc'], best_node_fm['auc']))
    print('n_pre_rec_fm: (%.4f,%.4f,%.4f) -- (%.4f,%.4f,%.4f) ' %
          (best_auc_re['n_pre_rec_fm'][0],
           best_auc_re['n_pre_rec_fm'][1],
           best_auc_re['n_pre_rec_fm'][2],
           best_node_fm['n_pre_rec_fm'][0],
           best_node_fm['n_pre_rec_fm'][1],
           best_node_fm['n_pre_rec_fm'][2]))
    print('para: ', best_auc_re['para'], best_node_fm['para'])
    print('-' * 80)


def show_test01():
    f_name = root_output + 'results_breast_cancer_test01.pkl'
    results = pickle.load(open(f_name))
    sparsity_list = results[0]['sparsity_list']
    for i in range(5):
        print('-' * 69)
        print('-' * 30 + 'folding %d' % i + '-' * 30)
        print('perf  : %.4f    %.4f    %.4f    %.4f' %
              (results[i]['iht']['perf'], results[i]['sto-iht']['perf'],
               results[i]['graph-iht']['perf'],
               results[i]['graph-sto-iht']['perf']))
        s1 = results[i]['iht']['s_star']
        s2 = results[i]['sto-iht']['s_star']
        s3 = results[i]['graph-iht']['s_star']
        s4 = results[i]['graph-sto-iht']['s_star']
        print('auc   : %.4f    %.4f    %.4f    %.4f' %
              (results[i]['iht']['auc'][s1],
               results[i]['sto-iht']['auc'][s2],
               results[i]['graph-iht']['auc'][s3],
               results[i]['graph-sto-iht']['auc'][s4]))
        s1, s2 = sparsity_list[s1], sparsity_list[s2]
        s3, s4 = sparsity_list[s3], sparsity_list[s4]
        print('best_s: %02d        %02d        %02d        %02d' %
              (s1, s2, s3, s4))
    exit()
    import matplotlib.pyplot as plt
    method_list = ['batch', 'sto', 'lr', 'overlap']
    averaged_results = None
    color_list = ['b', 'r', 'y', 'g', 'm', 'c', 'k']
    marker_list = ['s', 'D', '*', '>', '<', '+']
    fig, ax = plt.subplots(2, 3)
    for method_ind, method in enumerate(method_list):
        re = [averaged_results[method][i][0] for i in range(10)]
        ax[0, 0].plot(re, label=method, color=color_list[method_ind],
                      marker=marker_list[method_ind])
        ax[0, 0].legend()
        re = [averaged_results[method][i][1] for i in range(10)]
        ax[0, 1].plot(re, label=method, color=color_list[method_ind],
                      marker=marker_list[method_ind])
        ax[0, 1].legend()
        re = [averaged_results[method][i][2] for i in range(10)]
        ax[0, 2].plot(re, label=method, color=color_list[method_ind],
                      marker=marker_list[method_ind])
        ax[0, 2].legend()
        re = [averaged_results[method][i][3] for i in range(10)]
        ax[1, 0].plot(re, label=method, color=color_list[method_ind],
                      marker=marker_list[method_ind])
        ax[1, 0].legend()
        re = [averaged_results[method][i][4] for i in range(10)]
        ax[1, 1].plot(re, label=method, color=color_list[method_ind],
                      marker=marker_list[method_ind])
        ax[1, 1].legend()

    ax[0, 0].set_title('Node-Precision')
    ax[0, 1].set_title('Node-Recall')
    ax[0, 2].set_title('Node-F1-Score')
    ax[1, 0].set_title('AUC')
    ax[1, 1].set_title('ACC')
    plt.show()


def get_single_data(trial_i):
    import scipy.io as sio
    data = dict()
    f_name = 'results_bc_overlap_lasso_%02d.mat' % trial_i
    dataset = sio.loadmat(root_output + f_name)
    data['data'] = dataset['data'][0][0]
    # processing data sets
    re = {fold_i: dict() for fold_i in range(5)}
    for fold_ind, fold_i in enumerate(range(5)):
        re[fold_i]['x'] = np.asarray(data['data']['XX'])
        re[fold_i]['y'] = np.asarray([_[0] for _ in data['data']['Y']],
                                     dtype=float)
        re[fold_i]['entrez'] = [_[0] for _ in data['data']['entrez']]
        edges = np.asarray(data['data']['edges'], dtype=int)
        edges -= 1  # all indices are starts from zeros.
        re[fold_i]['edges'] = edges
        re_ii = dict()
        for i in range(5):
            re_ii[i] = data['data']['splitsstr'][0][0]['splits'][0][i]
        re[fold_i]['splits'] = re_ii
        re_ii = {i: dict() for i in range(5)}
        for i in range(5):
            xx = re[fold_i]['splits'][0][0]['train'][0][0]
            re_ii[i]['train'] = [_ - 1 for _ in xx]
            xx = re[fold_i]['splits'][0][0]['test'][0][0]
            re_ii[i]['test'] = [_ - 1 for _ in xx]
        re[fold_i]['splits'] = re_ii

        re_ii = dict()
        for i in range(5):
            re_ii[i] = data['data']['splitsstr'][0][0]['subsplits'][0][i]
            re_jj = {xx: dict() for xx in range(5)}
            for j in range(5):
                xx = re_ii[i][0][j][0][0]['train'][0]
                re_jj[j]['train'] = [_ - 1 for _ in xx]
                xx = re_ii[i][0][j][0][0]['test'][0]
                re_jj[j]['test'] = [_ - 1 for _ in xx]
            re_ii[i] = re_jj
        re[fold_i]['subsplits'] = re_ii
        re[fold_i]['groups'] = data['data']['groups']
    data['data'] = re
    # processing results
    # do not mismatch the order !!
    data['re_graph_lasso'] = dataset['res'][0][0]
    data['re_graph_overlap'] = dataset['res'][0][1]
    data['re_path_lasso'] = dataset['res'][0][2]
    data['re_path_overlap'] = dataset['res'][0][3]

    data['re_graph_lasso'] = [data['re_graph_lasso'][0][i][0][0]
                              for i in range(5)]
    data['re_graph_overlap'] = [data['re_graph_overlap'][0][i][0][0]
                                for i in range(5)]
    data['re_path_lasso'] = [data['re_path_lasso'][0][i][0][0]
                             for i in range(5)]
    data['re_path_overlap'] = [data['re_path_overlap'][0][i][0][0]
                               for i in range(5)]
    for method in ['re_graph_lasso', 're_graph_overlap',
                   're_path_lasso', 're_path_overlap']:
        re = {fold_i: dict() for fold_i in range(5)}
        for fold_ind, fold_i in enumerate(range(5)):
            lambdas = data[method][fold_i]['lambdas'][0]
            re[fold_i]['lambdas'] = data[method][fold_i]['lambdas'][0]
            re[fold_i]['kidx'] = data[method][fold_i]['kidx'][0]
            re[fold_i]['kgroups'] = data[method][fold_i]['kgroups'][0]
            re[fold_i]['kgroupidx'] = data[method][fold_i]['kgroupidx'][0]
            re[fold_i]['groups'] = data[method][fold_i]['groups'][0]
            re[fold_i]['sbacc'] = data[method][fold_i]['sbacc'][0]
            re[fold_i]['AS'] = data[method][fold_i]['AS'][0]
            re[fold_i]['completeAS'] = data[method][fold_i]['completeAS'][0]
            re[fold_i]['lstar'] = data[method][fold_i]['lstar'][0][0]
            ind_lstar = np.where(lambdas == re[fold_i]['lstar'])[0][0]
            re[fold_i]['auc'] = data[method][fold_i]['auc'][0][ind_lstar]
            re[fold_i]['acc'] = data[method][fold_i]['acc'][0]
            re[fold_i]['bacc'] = data[method][fold_i]['bacc'][0]
            re[fold_i]['perf'] = data[method][fold_i]['perf'][0][0]
            re[fold_i]['pred'] = data[method][fold_i]['pred']
            re[fold_i]['Ws'] = data[method][fold_i]['Ws']
            re[fold_i]['oWs'] = data[method][fold_i]['oWs']
            re[fold_i]['nextGrad'] = data[method][fold_i]['nextGrad'][0]
        data[method] = re
    return data


def load_dataset():
    method_list = ['re_graph_lasso', 're_graph_overlap', 're_path_lasso',
                   're_path_overlap']
    num_folding = 9
    all_data = {i: get_single_data(i) for i in range(num_folding)}
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


def get_single_graph_data(trial_i):
    f_name = root_output + 'iter_20/results_bc_graph-sto-iht_%02d.pkl' % trial_i
    method_list2 = ['iht', 'sto-iht', 'graph-iht', 'graph-sto-iht']
    results = pickle.load(open(f_name))
    for folding_i in range(5):
        for method in method_list2:
            re = results[folding_i][method]
            results[folding_i][method]['auc'] = re['auc'][re['s_star']]
    return results


def show_test05():
    method_list = ['re_graph_lasso', 're_graph_overlap', 're_path_lasso',
                   're_path_overlap']
    method_list2 = ['iht', 'sto-iht', 'graph-iht', 'graph-sto-iht']
    folding_list = range(20)
    all_data = {i: get_single_data(i) for i in folding_list}
    g_data = {i: get_single_graph_data(i) for i in folding_list}
    print('-' * 126)
    print('-' * 60 + ' bacc ' + '-' * 60)
    print('                    lasso-1      overlap-1    lasso-2      '
          'overlap-2    iht          sto-iht      graph-iht     graph-sto-iht')
    summary_results = {method: [] for method in method_list}
    summary_results2 = {method: [] for method in method_list2}
    for folding_i in folding_list:
        re = {method: [0.0, 0.0] for method in method_list}
        for method in method_list:
            perf_list = [all_data[folding_i][method][fold_i]['auc']
                         for fold_i in range(5)]
            summary_results[method].extend(perf_list)
            perf_list = np.asarray(perf_list)
            re[method][0] = np.mean(perf_list)
            re[method][1] = np.std(perf_list)
        re2 = {method: [0.0, 0.0] for method in method_list2}
        for method in method_list2:
            perf_list = [g_data[folding_i][fold_i][method]['auc']
                         for fold_i in range(5)]
            summary_results2[method].extend(perf_list)
            perf_list = np.asarray(perf_list)
            re2[method][0] = np.mean(perf_list)
            re2[method][1] = np.std(perf_list)
        print('%02d & '
              '%.3f,%.3f & %.3f,%.3f & %.3f,%.3f & %.3f,%.3f  '
              '%.3f,%.3f & %.3f,%.3f & %.3f,%.3f &  %.3f,%.3f \\\\' % (
                  folding_i + 1,
                  re['re_graph_lasso'][0], re['re_graph_lasso'][1],
                  re['re_graph_overlap'][0], re['re_graph_overlap'][1],
                  re['re_path_lasso'][0], re['re_path_lasso'][1],
                  re['re_path_overlap'][0], re['re_path_overlap'][1],
                  re2['iht'][0], re2['iht'][1],
                  re2['sto-iht'][0], re2['sto-iht'][1],
                  re2['graph-iht'][0], re2['graph-iht'][1],
                  re2['graph-sto-iht'][0], re2['graph-sto-iht'][1]))
    print('- & '
          '%.3f,%.3f & %.3f,%.3f & %.3f,%.3f & %.3f,%.3f  '
          '%.3f,%.3f & %.3f,%.3f & %.3f,%.3f &  %.3f,%.3f' % (
              float(np.mean(summary_results['re_graph_lasso'])),
              float(np.std(summary_results['re_graph_lasso'])),
              float(np.mean(summary_results['re_graph_overlap'])),
              float(np.std(summary_results['re_graph_overlap'])),
              float(np.mean(summary_results['re_path_lasso'])),
              float(np.std(summary_results['re_path_lasso'])),
              float(np.mean(summary_results['re_path_overlap'])),
              float(np.std(summary_results['re_path_overlap'])),

              float(np.mean(summary_results2['iht'])),
              float(np.std(summary_results2['iht'])),
              float(np.mean(summary_results2['sto-iht'])),
              float(np.std(summary_results2['sto-iht'])),
              float(np.mean(summary_results2['graph-iht'])),
              float(np.std(summary_results2['graph-iht'])),
              float(np.mean(summary_results2['graph-sto-iht'])),
              float(np.std(summary_results2['graph-sto-iht']))))
    print('-' * 126)


def show_test04():
    method_list = ['re_graph_lasso', 're_graph_overlap', 're_path_lasso',
                   're_path_overlap']
    method_list2 = ['iht', 'sto-iht', 'graph-iht', 'graph-sto-iht']
    folding_list = range(20)
    all_data = {i: get_single_data(i) for i in folding_list}
    g_data = {i: get_single_graph_data(i) for i in folding_list}
    print('-' * 126)
    print('-' * 60 + ' bacc ' + '-' * 60)
    print('                    lasso-1      overlap-1    lasso-2      '
          'overlap-2    iht          sto-iht      graph-iht     graph-sto-iht')
    summary_results = {method: [] for method in method_list}
    summary_results2 = {method: [] for method in method_list2}
    for folding_i in folding_list:
        re = {method: [0.0, 0.0] for method in method_list}
        for method in method_list:
            perf_list = [all_data[folding_i][method][fold_i]['auc']
                         for fold_i in range(5)]
            summary_results[method].extend(perf_list)
            perf_list = np.asarray(perf_list)
            re[method][0] = np.mean(perf_list)
            re[method][1] = np.std(perf_list)
        re2 = {method: [0.0, 0.0] for method in method_list2}
        for method in method_list2:
            perf_list = [g_data[folding_i][fold_i][method]['auc']
                         for fold_i in range(5)]
            summary_results2[method].extend(perf_list)
            perf_list = np.asarray(perf_list)
            re2[method][0] = np.mean(perf_list)
            re2[method][1] = np.std(perf_list)
        print('%02d & '
              '%.3f,%.3f & %.3f,%.3f & %.3f,%.3f & %.3f,%.3f  '
              '%.3f,%.3f & %.3f,%.3f & %.3f,%.3f &  %.3f,%.3f \\\\' % (
                  folding_i + 1,
                  re['re_graph_lasso'][0], re['re_graph_lasso'][1],
                  re['re_graph_overlap'][0], re['re_graph_overlap'][1],
                  re['re_path_lasso'][0], re['re_path_lasso'][1],
                  re['re_path_overlap'][0], re['re_path_overlap'][1],
                  re2['iht'][0], re2['iht'][1],
                  re2['sto-iht'][0], re2['sto-iht'][1],
                  re2['graph-iht'][0], re2['graph-iht'][1],
                  re2['graph-sto-iht'][0], re2['graph-sto-iht'][1]))
    print('- & '
          '%.3f,%.3f & %.3f,%.3f & %.3f,%.3f & %.3f,%.3f  '
          '%.3f,%.3f & %.3f,%.3f & %.3f,%.3f &  %.3f,%.3f' % (
              float(np.mean(summary_results['re_graph_lasso'])),
              float(np.std(summary_results['re_graph_lasso'])),
              float(np.mean(summary_results['re_graph_overlap'])),
              float(np.std(summary_results['re_graph_overlap'])),
              float(np.mean(summary_results['re_path_lasso'])),
              float(np.std(summary_results['re_path_lasso'])),
              float(np.mean(summary_results['re_path_overlap'])),
              float(np.std(summary_results['re_path_overlap'])),

              float(np.mean(summary_results2['iht'])),
              float(np.std(summary_results2['iht'])),
              float(np.mean(summary_results2['sto-iht'])),
              float(np.std(summary_results2['sto-iht'])),
              float(np.mean(summary_results2['graph-iht'])),
              float(np.std(summary_results2['graph-iht'])),
              float(np.mean(summary_results2['graph-sto-iht'])),
              float(np.std(summary_results2['graph-sto-iht']))))
    print('-' * 126)


def main():
    show_test05()


if __name__ == "__main__":
    main()
