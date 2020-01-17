import pickle
import numpy as np


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


def show_genes_convex():
    show_genes = dict()
    for i in range(20):
        data = get_single_data(trial_i=0, root_input='data/')
        show_genes['data_entrez'] = data['data_entrez']
        show_genes['cancer_related_genes'] = data['cancer_related_genes']
        show_genes[i] = {_: dict() for _ in data['found_related_genes']}
        for method in data['found_related_genes']:
            print(i, method)
            nodes, edges = dict(), dict()
            for fold_i in data[method]:
                show_genes[i][method][fold_i] = dict()
                kidx = data[method][fold_i]['kidx']
                index = list(data[method][fold_i]['lambdas']).index(data[method][fold_i]['lstar'])
                w = data[method][fold_i]['oWs'][:, index][1:]
                found_genes = {_: '' for _ in kidx[np.nonzero(w)[0]]}
                for gene in found_genes:
                    nodes[data['data_entrez'][gene]] = ''
            for edge in data['data_edges']:
                if edge[0] in nodes and edge[1] in nodes:
                    edges[(data['data_entrez'][edge[0]], data['data_entrez'][edge[1]])] = ''
            show_genes[i][method]['nodes'] = nodes
            show_genes[i][method]['edges'] = edges
    pickle.dump(show_genes, open('data/show_genes_convex.pkl', 'wb'))


def show_genes_nonconvex(data_entrez):
    method_list = ['iht', 'sto-iht', 'graph-iht', 'graph-sto-iht']
    folding_list = range(20)
    max_epochs = 40
    show_genes = dict()
    cancer_related_genes = {
        4288: 'MKI67', 1026: 'CDKN1A', 472: 'ATM', 7033: 'TFF3', 2203: 'FBP1', 7494: 'XBP1', 1824: 'DSC2',
        1001: 'CDH3', 11200: 'CHEK2', 7153: 'TOP2A', 672: 'BRCA1', 675: 'BRCA2', 580: 'BARD1', 9: 'NAT1',
        771: 'CA12', 367: 'AR', 7084: 'TK2', 5892: 'RAD51D', 2625: 'GATA3', 7155: 'TOP2B', 896: 'CCND3', 894: 'CCND2',
        10551: 'AGR2', 3169: 'FOXA1', 2296: 'FOXC1'}
    show_genes['cancer_related_genes'] = cancer_related_genes
    import csv
    ori_edges = []
    with open('data/edge.txt') as csvfile:
        for row in csv.reader(csvfile, delimiter='\t', quotechar='|'):
            ori_edges.append((data_entrez[int(row[0]) - 1], data_entrez[int(row[1]) - 1]))
    for trial_i in folding_list:
        show_genes[trial_i] = {_: dict() for _ in method_list}
        f_name = 'results/results_exp_bc_%02d_%02d.pkl' % (trial_i, max_epochs)
        data = pickle.load(open(f_name))
        for method in method_list:
            print(trial_i, method)
            show_genes[trial_i][method] = dict()
            nodes, edges = dict(), dict()
            for fold_i in data:
                wt = data[fold_i][method]['w_hat']
                for node in np.nonzero(wt[:len(wt) - 1])[0]:
                    id_ = data[fold_i][method]['map_entrez'][node]
                    nodes[id_] = ''
            for edge in ori_edges:
                if edge[0] in nodes and edge[1] in nodes:
                    edges[edge] = ''
            show_genes[trial_i][method]['nodes'] = list(nodes.keys())
            show_genes[trial_i][method]['edges'] = list(edges.keys())
    pickle.dump(show_genes, open('results/show_genes_nonconvex.pkl', 'wb'))


def show_detected_genes():
    import networkx as nx
    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = 8, 8.
    from matplotlib import rc
    from matplotlib.font_manager import FontProperties
    plt.rc('font', **{'size': 8, 'weight': 'bold'})
    rc('text', usetex=True)
    font0 = FontProperties()
    font0.set_weight('bold')
    method_list = ['re_edge_re_lasso', 're_path_re_lasso', 're_edge_re_overlap', 're_path_re_lasso',
                   'iht', 'sto-iht', 'graph-iht', 'graph-sto-iht']
    label_list = [r'\textsc{Edge-L1}', r'\textsc{Path-L1}', r'\textsc{Edge-Over}',
                  r'\textsc{Path-Over}',
                  r'\textsc{IHT}', r'\textsc{StoIHT}',
                  r'\textsc{GraphIHT}', r'\textsc{GraphSIHT}']
    all_data = dict()
    data_convex = pickle.load(open('data/show_genes_convex.pkl'))
    all_data['cancer_related_genes'] = data_convex['cancer_related_genes']
    all_data['data_entrez'] = data_convex['data_entrez']
    data_nonconvex = pickle.load(open('results/show_genes_nonconvex.pkl'))
    for i in range(20):
        all_data[i] = dict()
        for method in data_convex[i]:
            all_data[i][method] = data_convex[i][method]
        for method in data_nonconvex[i]:
            all_data[i][method] = data_nonconvex[i][method]
    true_nodes = all_data['cancer_related_genes'].keys()
    fig, ax = plt.subplots(5, 8)
    for trial_i_ind, trial_i in enumerate(range(0, 5)):
        for method_ind, method in enumerate(method_list):
            print(trial_i, method)
            g = nx.Graph()
            detected_edges = all_data[trial_i_ind][method]['edges']
            detected_nodes = all_data[trial_i_ind][method]['nodes']
            intersect = list(set(true_nodes).intersection(detected_nodes))
            for edge_ind, edge in enumerate(detected_edges):
                g.add_edge(edge[0], edge[1])
            for node in list(detected_nodes):
                g.add_node(node)
            for node in list(true_nodes):
                g.add_node(node)
            g = nx.minimum_spanning_tree(g)
            print('method: %s pre: %.3f rec: %.3f fm: %.3f' % (
                method, float(len(intersect)) / float(len(detected_nodes)),
                float(len(intersect)) / float(len(true_nodes)),
                float(2.0 * len(intersect)) / float(len(detected_nodes) + len(true_nodes))))
            color_list = []
            for node in detected_nodes:
                if node in intersect:
                    color_list.append('r')
                else:
                    color_list.append('b')
            print(nx.nodes(g))
            nx.draw_spring(g, ax=ax[trial_i_ind, method_ind], node_size=10, edge_color='black',
                           edge_width=2., font_size=4, node_edgecolor='black',
                           node_facecolor='white', node_edgewidth=1., k=10.0,
                           nodelist=detected_nodes, node_color=color_list)
            ax[trial_i_ind, method_ind].axis('on')
            ax[0, method_ind].set(title='%s' % label_list[method_ind])
            plt.setp(ax[trial_i_ind, method_ind].get_xticklabels(), visible=False)
            plt.setp(ax[trial_i_ind, method_ind].get_yticklabels(), visible=False)
            ax[trial_i_ind, method_ind].tick_params(axis='both', which='both', length=0)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    f_name = 'bc_figures_genes_all_0_5.pdf'
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0.03, format='pdf')
    plt.close()


def main():
    show_detected_genes()


if __name__ == '__main__':
    main()
