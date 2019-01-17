# -*- coding: utf-8 -*-
import random
import numpy as np

__all__ = ['simu_grid_graph', 'random_walk', 'expit', 'logistic_predict',
           'logit_loss_bl', 'node_pre_rec_fm', 'least_square_predict',
           'logit_loss_grad_bl', 'logit_loss_grad', 'sensing_matrix']


def simu_grid_graph(width, height, rand_weight=False):
    """Generate a grid graph.
    To generate a grid graph. Each node has 4-neighbors. Please see more
    details in https://en.wikipedia.org/wiki/Lattice_graph. For example,
    we can generate 5x3(width x height) grid graph
                0---1---2---3---4
                |   |   |   |   |
                5---6---7---8---9
                |   |   |   |   |
                10--11--12--13--14
    by using simu_grid_graph(5, 3)
    We can also generate a 1x5 chain graph
                0---1---2---3---4
    by using simu_grid_graph(5, 1)
    :param width: width of this grid graph.
    :param height: height of this grid graph.
    :param rand_weight: generate weights from U(1., 2.) if it is True.
    :return: edges and corresponding edge costs.
    return two empty [],[] list if there was any error occurring.
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


def random_walk(edges, s, init_node=None, restart=0.0):
    """Random generate a connected subgraph by using random walk.
    Given a connected undirected graph (represented as @param:edges), a random
    walk is a procedure to generate a connected subgraph with s different
    nodes. Please check more details in the first paragraph of section 1.
    basic notations and facts of reference [1] in Page 3.
    Reference:  [1] Lovász, László. "Random walks on graphs: A survey."
                    Combinatorics, Paul erdos is eighty 2.1 (1993): 1-46.
    :param edges: input graph as the list of edges.
    :param s: the number of nodes in the returned subgraph.
    :param init_node: initial point of the random walk.
    :param restart: with a fix probability to restart from the initial node.
    :return: a list of s nodes and a list of walked edges.
    return two empty list if there was any error occurring.
    """
    np.random.seed()
    adj, nodes = dict(), set()
    for edge in edges:  # construct the adjacency matrix.
        uu, vv = int(edge[0]), int(edge[1])
        nodes.add(uu)
        nodes.add(vv)
        if uu not in adj:
            adj[uu] = set()
        adj[uu].add(vv)
        if vv not in adj:
            adj[vv] = set()
        adj[vv].add(uu)
    if init_node is None:
        # random select an initial node.
        rand_start_point = random.choice(list(nodes))
        init_node = list(adj.keys())[rand_start_point]
    if init_node not in nodes:
        print('Error: the initial_node is not in the graph!')
        return [], []
    if not (0.0 <= restart < 1.0):
        print('Error: the restart probability not in (0.0,1.0)')
        return [], []
    if not (0 <= s <= len(nodes)):
        print('Error: the number of nodes not in [0,%d]' % len(nodes))
        return [], []
    subgraph_nodes, subgraph_edges = set(), set()
    next_node = init_node
    subgraph_nodes.add(init_node)
    if s <= 1:
        return subgraph_nodes, subgraph_edges
    # get a connected subgraph with s nodes.
    while len(subgraph_nodes) < s:
        next_neighbors = list(adj[next_node])
        rand_nei = random.choice(next_neighbors)
        subgraph_nodes.add(rand_nei)
        subgraph_edges.add((next_node, rand_nei))
        subgraph_edges.add((rand_nei, next_node))
        next_node = rand_nei  # go to next node.
        if random.random() < restart:
            next_node = init_node
    return list(subgraph_nodes), list(subgraph_edges)


def sensing_matrix(n, x, norm_noise=0.0):
    """ Generate a Gaussian matrix and corresponding n measurements.
    Please see equation 1.2 in [1]
    Reference:
        [1] Needell, Deanna, and Joel A. Tropp. "CoSaMP: Iterative signal
            recovery from incomplete and inaccurate samples."
            Applied and computational harmonic analysis 26.3 (2009): 301-321.
    :param n: the number of measurements need to sensing.
    :param x: true signal.
    :param norm_noise: add noise by using: ||e|| = norm_noise.
    :return:
        x_mat: sensing matrix
        y_tr: measurement vector.
        y_e: measurement vector + ||e||
    """
    p = len(x)
    x_mat = np.random.normal(0.0, 1.0, size=(n * p)) / np.sqrt(n)
    x_mat = x_mat.reshape((n, p))
    y_tr = np.dot(x_mat, x)
    noise_e = np.random.normal(0.0, 1.0, len(y_tr))
    y_e = y_tr + (norm_noise / np.linalg.norm(noise_e)) * noise_e
    return x_mat, y_tr, y_e


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


def logistic(x):
    """
    logistic is also known as expit. Please see expit.
    :param x: np.ndarray
    :return:
    """
    return expit(x)


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


def least_square_predict(x_va, wt):
    """ To predict the probability for sample xi. """
    pred_val, p = [], x_va.shape[1]
    for i in range(len(x_va)):
        pred_val.append(np.dot(wt[:p], x_va[i] + wt[p]))
    return np.asarray(pred_val)


def log_logistic(x):
    """ return log( 1/(1+exp(-x)) )"""
    out = np.zeros_like(x)
    posi = np.where(x > 0.0)
    nega = np.where(x <= 0.0)
    out[posi] = -np.log(1. + np.exp(-x[posi]))
    out[nega] = x[nega] - np.log(1. + np.exp(x[nega]))
    return out


def _grad_w(x_tr, y_tr, wt, eta):
    """ return {+1,-1} Logistic (val,grad) on training samples. """
    assert len(wt) == (x_tr.shape[1] + 1)
    c, p = wt[-1], x_tr.shape[1]
    wt = wt[:p]
    yz = y_tr * (np.dot(x_tr, wt) + c)
    z = expit(yz)
    loss = -np.sum(log_logistic(yz)) + .5 * eta * np.dot(wt, wt)
    grad = np.zeros(p + 1)
    z0 = (z - 1) * y_tr
    grad[:p] = np.dot(x_tr.T, z0) + eta * wt
    grad[-1] = z0.sum()
    return loss, grad


def logit_loss_grad(x_tr, y_tr, wt, eta):
    """ return {+1,-1} Logistic (val,grad) on training samples. """
    assert len(wt) == (x_tr.shape[1] + 1)
    c, n, p = wt[-1], x_tr.shape[0], x_tr.shape[1]
    wt = wt[:p]
    yz = y_tr * (np.dot(x_tr, wt) + c)
    z = expit(yz)
    loss = -np.sum(log_logistic(yz)) + .5 * eta * np.dot(wt, wt)
    grad = np.zeros(p + 1)
    z0 = (z - 1) * y_tr
    grad[:p] = np.dot(x_tr.T, z0) + eta * wt
    grad[-1] = z0.sum()
    return loss / float(n), grad / float(n)


def logit_loss_bl(x_tr, y_tr, wt, l2_reg, cp, cn):
    """
    Calculate the balanced loss and gradient of the logistic function.
    :param x_tr: (n,p), where p is the number of features.
    :param y_tr: (n,), where n is the number of labels.
    :param wt: current model. wt[-1] is the intercept.
    :param l2_reg: regularization to avoid overfitting.
    :param cp:
    :param cn:
    :return:
    """
    """ return {+1,-1} Logistic (val,grad) on training samples. """
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


def logit_loss_grad_bl(x_tr, y_tr, wt, l2_reg, cp, cn):
    """
    Calculate the balanced loss and gradient of the logistic function.
    :param x_tr: (n,p), where p is the number of features.
    :param y_tr: (n,), where n is the number of labels.
    :param wt: current model. wt[-1] is the intercept.
    :param l2_reg: regularization to avoid overfitting.
    :param cp:
    :param cn:
    :return:
    """
    """ return {+1,-1} Logistic (val,grad) on training samples. """
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


def node_pre_rec_fm(true_nodes, pred_nodes):
    """ Return the precision, recall and f-measure.
    :param true_nodes:
    :param pred_nodes:
    :return: precision, recall and f-measure """
    true_nodes, pred_nodes = set(true_nodes), set(pred_nodes)
    pre, rec, fm = 0.0, 0.0, 0.0
    if len(pred_nodes) != 0:
        pre = len(true_nodes & pred_nodes) / float(len(pred_nodes))
    if len(true_nodes) != 0:
        rec = len(true_nodes & pred_nodes) / float(len(true_nodes))
    if (pre + rec) > 0.:
        fm = (2. * pre * rec) / (pre + rec)
    return [pre, rec, fm]


def auc_node_fm(auc, node_fm):
    if 0.0 <= auc <= 1.0 and 0.0 <= node_fm <= 1.0:
        return 2.0 * (auc * node_fm) / (auc + node_fm)
    else:
        print('auc and node-fm must be in the range [0.0,1.0]')
        exit(0)


def m_print(result, method, trial_i, n_tr_, fig_i, mu, sub_graph,
            header=False):
    if header:
        print('-' * 165)
        print('method         fig_i  s      tr_id '
              ' n_tr       mu   auc      acc     f1      ' +
              'n_pre   n_rec   n_fm     nega_in  nega_out'
              '  posi_in  posi_out intercept  run_time')
    auc = result['auc'][-1]
    acc = result['acc'][-1]
    f1 = result['f1'][-1]
    node_pre = result['n_pre'][-1]
    node_rec = result['n_rec'][-1]
    node_fm = result['n_fm'][-1]
    num_nega_in = len([_ for ind, _ in enumerate(result['wt'][-1]) if
                       ind in sub_graph and _ < 0.0])
    num_nega_out = len([_ for ind, _ in enumerate(result['wt'][-1]) if
                        ind not in sub_graph and _ < 0.0])
    num_posi_in = len([_ for ind, _ in enumerate(result['wt'][-1]) if
                       ind in sub_graph and _ > 0.0])
    num_posi_out = len([_ for ind, _ in enumerate(result['wt'][-1]) if
                        ind not in sub_graph and _ > 0.0])
    sparsity = np.count_nonzero(result['wt'][-1][:1089])
    intercept = result['intercept'][-1]
    run_time = result['run_time'][-1]
    print('{:14s} {:6s} {:6s} {:6s} {:6s} {:7.1f} '
          '{:7.4f}  {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} '
          '{:8d} {:8d} {:8d} {:8d} {:12.4f} {:12.3f}'
          .format(method, fig_i, str(sparsity), str(trial_i), str(n_tr_),
                  mu, auc, acc, f1, node_pre, node_rec, node_fm, num_nega_in,
                  num_nega_out, num_posi_in, num_posi_out, intercept,
                  run_time))


def gen_test_case(x_tr, y_tr, w0, edges, weights):
    f = open('test_case.txt', 'wb')
    f.write('P %d %d %d\n' % (len(x_tr), len(x_tr[0]), len(edges)))
    for i in range(len(x_tr)):
        f.write('x_tr ')
        for j in range(len(x_tr[i])):
            f.write('%.8f' % x_tr[i][j] + ' ')
        f.write(str(y_tr[i]) + '\n')
    for i in range(len(edges)):
        f.write('E ' + str(edges[i][0]) + ' ' +
                str(edges[i][1]) + ' ' + '%.8f' % weights[i] + '\n')
    for i in range(len(w0)):
        f.write('N %d %.8f\n' % (i, w0[i]))
    f.close()


def test_expit():
    print(expit(np.asarray([0.0])))
    print(expit(np.asarray([-1.0, 1.0])))
    print(expit(np.asarray([-10.0, 10.0])))
    print(expit(np.asarray([-1e5, 1e5])))


def test_logistic():
    x = np.asarray([[0.1, 0.2], [1., 1.], [0., 0.], [-1., -1.]])
    w = np.asarray([-0.1, 1.0, 0.0])
    print('predicted probability: '),
    print(logistic_predict(x, w)[0])
    print('predicted labels: '),
    print(logistic_predict(x, w)[1])


def test_random_walk():
    edges, costs = simu_grid_graph(4, 4)
    nodes, edge_list = random_walk(edges=edges, s=5, init_node=8)
    print(nodes)
    print(edge_list)


def main():
    edges, costs = simu_grid_graph(5, 3)
    print(edges)
    print(costs)
    exit()
    test_random_walk()
    test_logistic()
    test_expit()


if __name__ == '__main__':
    main()
