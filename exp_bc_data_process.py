# -*- coding: utf-8 -*-
import os
import csv
import pickle
import numpy as np


def get_related_genes():
    gen_list_dict_01 = {675: ['BRCA2'],
                        672: ['BRCA1']}
    # found in the following paper:
    # [1] Agius, Phaedra, Yiming Ying, and Colin Campbell. "Bayesian
    # unsupervised learning with multiple data types." Statistical
    # applications in genetics and molecular biology 8.1 (2009): 1-27.
    gen_list_dict_02 = {2568: ['GABRP'],
                        1824: ['DSC2'],
                        9: ['NAT1'],
                        834139: ['XPB1'],
                        10551: ['AGR2'],
                        771: ['CA12'],
                        7033: ['TFF3'],
                        3169: ['FOXA1'],
                        367: ['AR'],
                        2203: ['FBP1'],
                        1001: ['CDH3'],
                        2625: ['GATA3'],
                        2296: ['FOXC1'],
                        7494: ['XBP1']}

    # [2] Couch, Fergus J., et al. "Associations between
    # cancer predisposition testing panel genes and breast cancer."
    # JAMA oncology 3.9 (2017): 1190-1196.
    gen_list_dict_03 = {3169: ['FOXA1'],
                        472: ['ATM'],
                        580: ['BARD1'],
                        11200: ['CHEK2'],
                        79728: ['PALB2'],
                        5892: ['RAD51D']}
    # [3] Rheinbay, Esther, et al. "Recurrent and functional regulatory
    # mutations in breast cancer." Nature 547.7661 (2017): 55-60.
    gen_list_dict_04 = {3169: ['FOXA1'],
                        6023: ['RMRP', 'CHH', 'NME1', 'RRP2', 'RMRPR'],
                        283131: ['NEAT1']}
    # [4] Györffy, Balazs, et al. "An online survival analysis tool to rapidly
    #  assess the effect of 22,277 genes on breast cancer prognosis using
    # microarray data of 1,809 patients." Breast cancer research and
    # treatment 123.3 (2010): 725-731.
    gen_list_dict_05 = {7153: ['TOP2A'],
                        7155: ['TOP2B'],
                        4288: ['MKI67'],
                        894: ['CCND2'],
                        896: ['CCND3'],
                        1026: ['CDKN1A'],
                        7084: ['TK2']}
    all_related_genes = dict()
    for key in gen_list_dict_01:
        all_related_genes[key] = gen_list_dict_01[key]
    for key in gen_list_dict_02:
        all_related_genes[key] = gen_list_dict_02[key]
    for key in gen_list_dict_03:
        all_related_genes[key] = gen_list_dict_03[key]
    for key in gen_list_dict_04:
        all_related_genes[key] = gen_list_dict_04[key]
    for key in gen_list_dict_05:
        all_related_genes[key] = gen_list_dict_05[key]
    return all_related_genes


def raw_data_process(root_input):
    import scipy.io as sio
    import networkx as nx
    raw = sio.loadmat(root_p + 'raw/vant.mat')
    data = {'x': np.asarray(raw['X']),
            'y': np.asarray([_[1] for _ in raw['Y']]),
            'entrez': [_[0] for _ in np.asarray(raw['entrez'])]}
    for i in range(len(data['x'][0])):
        if np.mean(data['x'][:, i]) == 0. and np.std(data['x'][:, i]) == 0.0:
            print('default values.')
    edges, costs = [], []
    g = nx.Graph()
    with open(root_p + 'raw/edge.txt') as csvfile:
        edge_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in edge_reader:
            g.add_edge(row[0], row[1])
            edges.append([int(row[0]) - 1, int(row[1]) - 1])
            costs.append(1.)
    pathways = dict()
    with open(root_p + 'raw/pathways.txt') as csvfile:
        edge_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in edge_reader:
            if int(row[0]) not in pathways:
                pathways[int(row[0])] = []
            pathways[int(row[0])].append(int(row[1]))
    print(nx.number_connected_components(g))
    print(nx.number_of_edges(g))
    print(nx.number_of_nodes(g))
    nodes = [int(_) for _ in nx.nodes(g)]
    print(min(nodes), max(nodes), len(nodes))
    data['edges'] = np.asarray(edges, dtype=int)
    data['costs'] = np.asarray(costs, dtype=np.float64)
    data['pathways'] = pathways

    # ------------- get entrez gene maps.
    test_entrez_gene_names = dict()
    with open(root_input + 'entrez_gene_map_from_match_miner.txt') as csvfile:
        line_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row_ind, row in enumerate(line_reader):
            if row_ind >= 21:
                if int(row[2]) not in test_entrez_gene_names:
                    test_entrez_gene_names[int(row[2])] = []
                test_entrez_gene_names[int(row[2])].append(row[3])

    entrez_gene_names = dict()
    with open(root_input + 'entrez_gene_map_from_uniprot.tab') as csvfile:
        line_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row_ind, row in enumerate(line_reader):
            if row_ind >= 1:
                if str.find(row[-1], ',') != -1:
                    for entrez_id in str.split(row[-1], ','):
                        gene_names = str.split(row[-2], ' ')
                        gene_names = [_ for _ in gene_names if len(_) > 0]
                        if int(entrez_id) not in entrez_gene_names:
                            entrez_gene_names[int(entrez_id)] = []
                        entrez_gene_names[int(entrez_id)].extend(gene_names)
                else:
                    entrez_id = int(row[-1])
                    gene_names = str.split(row[-2], ' ')
                    gene_names = [_ for _ in gene_names if len(_) > 0]
                    if int(entrez_id) not in entrez_gene_names:
                        entrez_gene_names[int(entrez_id)] = []
                    entrez_gene_names[int(entrez_id)].extend(gene_names)

    all_entrez_gene_names = dict()
    for key in test_entrez_gene_names:
        all_entrez_gene_names[key] = test_entrez_gene_names[key]
    for key in entrez_gene_names:
        all_entrez_gene_names[key].extend(entrez_gene_names[key])
    data['entrez_gene_name_map'] = all_entrez_gene_names

    # ---------------- get genes names from van_t_veer_2002 nature paper
    f_name = root_p + 'raw/van_t_veer_2002/Table_NKI_295_1.txt'
    print('load data from: %s' % f_name)
    with open(f_name) as csvfile:
        edge_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        all_lines = [row for row_ind, row in enumerate(edge_reader)]
        col_gene_name_list = \
            [row[1] for row_ind, row in enumerate(all_lines) if row_ind >= 2]
        data['van_t_veer_gene_name_list'] = col_gene_name_list

    # ----------------- get cancer related genes
    all_genes_from_database = dict()
    for entrez_id in data['entrez_gene_name_map']:
        for gene in data['entrez_gene_name_map'][entrez_id]:
            all_genes_from_database[gene] = entrez_id
    all_related_genes = get_related_genes()
    finalized_cancer_gene = dict()
    for entrez in all_related_genes:
        flag = False
        for gene in all_related_genes[entrez]:
            if entrez in data['entrez']:
                print('find: %s' % gene)
                finalized_cancer_gene[gene] = entrez
                flag = True
                break
        if not flag:
            print('cannot find: %s' % all_related_genes[entrez])
    print('all related genes: %d' % len(all_related_genes))
    data['cancer_related_genes'] = finalized_cancer_gene
    pickle.dump(data, open(root_input + 'overlap_data.pkl', 'wb'))


def generate_original_data(root_input):
    row_samples_list = []
    col_gene_substance_list = []
    col_gene_name_list = []
    data_log_ratio = np.zeros(shape=(24496, 295))
    data_log_ratio_err = np.zeros(shape=(24496, 295))
    data_p_value = np.zeros(shape=(24496, 295))
    data_intensity = np.zeros(shape=(24496, 295))
    data_flag = np.zeros(shape=(24496, 295))
    anchor_list = [0, 50, 100, 150, 200, 250]
    step_list = [50, 50, 50, 50, 50, 45]
    for table_i in range(1, 7):
        f_name = root_p + 'raw/van_t_veer_2002/Table_NKI_295_%d.txt' % table_i
        print('load data from: %s' % f_name)
        with open(f_name) as csvfile:
            edge_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            all_lines = [row for row_ind, row in enumerate(edge_reader)]
            if table_i == 1:
                col_gene_substance_list = \
                    [row[0] for row_ind, row in enumerate(all_lines)
                     if row_ind >= 2]
                col_gene_name_list = \
                    [row[1] for row_ind, row in enumerate(all_lines)
                     if row_ind >= 2]
            for row_ind, row in enumerate(all_lines):
                if row_ind >= 2:
                    ii = anchor_list[table_i - 1]
                    step = step_list[table_i - 1]
                    re = [_ for _ind, _ in enumerate(row[2:]) if _ind % 5 == 0]
                    re = [float(_) for _ in re]
                    data_log_ratio[row_ind - 2][ii:ii + step] = re
                    re = [_ for _ind, _ in enumerate(row[2:]) if _ind % 5 == 1]
                    re = [float(_) for _ in re]
                    data_log_ratio_err[row_ind - 2][ii:ii + step] = re
                    re = [_ for _ind, _ in enumerate(row[2:]) if _ind % 5 == 2]
                    re = [float(_) for _ in re]
                    data_p_value[row_ind - 2][ii:ii + step] = re
                    re = [_ for _ind, _ in enumerate(row[2:]) if _ind % 5 == 3]
                    re = [float(_) for _ in re]
                    data_intensity[row_ind - 2][ii:ii + step] = re
                    re = [_ for _ind, _ in enumerate(row[2:]) if _ind % 5 == 4]
                    re = [float(_) for _ in re]
                    data_flag[row_ind - 2][ii:ii + step] = re
    data = {'row_samples_list': row_samples_list,
            'col_gene_substance_list': col_gene_substance_list,
            'col_gene_name_list': col_gene_name_list,
            'data_log_ratio': data_log_ratio,
            'data_log_ratio_err': data_log_ratio_err,
            'data_p_value': data_p_value,
            'data_intensity': data_intensity,
            'data_flag': data_flag}
    f_name = root_input + 'original_data.pkl'
    pickle.dump(data, open(f_name, 'wb'))


def map_entrez_gene_name(root_input):
    test_entrez_gene_names = dict()
    with open(root_input + 'entrez_gene_map_from_match_miner.txt') as csvfile:
        line_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row_ind, row in enumerate(line_reader):
            if row_ind >= 21:
                if int(row[2]) not in test_entrez_gene_names:
                    test_entrez_gene_names[int(row[2])] = []
                test_entrez_gene_names[int(row[2])].append(row[3])

    overlap_data = pickle.load(open(root_input + 'overlap_data.pkl'))
    original_data = pickle.load(open(root_input + 'original_data.pkl'))
    entrez_gene_names = dict()
    final_results = dict()
    with open(root_input + 'entrez_gene_map_from_uniprot.tab') as csvfile:
        line_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row_ind, row in enumerate(line_reader):
            if row_ind >= 1:
                if str.find(row[-1], ',') != -1:
                    for entrez_id in str.split(row[-1], ','):
                        gene_names = str.split(row[-2], ' ')
                        gene_names = [_ for _ in gene_names if len(_) > 0]
                        if int(entrez_id) not in entrez_gene_names:
                            entrez_gene_names[int(entrez_id)] = []
                        entrez_gene_names[int(entrez_id)].extend(gene_names)
                else:
                    entrez_id = int(row[-1])
                    gene_names = str.split(row[-2], ' ')
                    gene_names = [_ for _ in gene_names if len(_) > 0]
                    if int(entrez_id) not in entrez_gene_names:
                        entrez_gene_names[int(entrez_id)] = []
                    entrez_gene_names[int(entrez_id)].extend(gene_names)

    all_entrez_gene_names = dict()
    for key in test_entrez_gene_names:
        all_entrez_gene_names[key] = test_entrez_gene_names[key]
    for key in entrez_gene_names:
        all_entrez_gene_names[key].extend(entrez_gene_names[key])

    index, num_gene_not_found, num_entrez_not_found = 0, 0, 0
    for item_ind, cur_entrez in enumerate(overlap_data['entrez']):
        if cur_entrez in all_entrez_gene_names:
            found_gene, ind = None, -1
            for each_gene in all_entrez_gene_names[cur_entrez]:
                if each_gene in original_data['col_gene_name_list']:
                    index += 1
                    re1 = overlap_data['x'][:, item_ind]
                    print('-' * 40)
                    print(len(np.where(re1 > 0.0)[0]),
                          len(np.where(re1 <= 0.0)[0]))
                    re2 = np.asarray(original_data['data_log_ratio'][item_ind])
                    print(len(np.where(re2 > 0.0)[0]),
                          len(np.where(re2 <= 0.0)[0]))
                    print('id: %d entrez: %d gene_name: %s' %
                          (index, cur_entrez, each_gene))
                    final_results[cur_entrez] = each_gene
                    found_gene = each_gene
                    break
            if found_gene is None:
                final_results[cur_entrez] = 'No_Gene_Found'
                num_gene_not_found += 1
        else:
            print(cur_entrez)
            num_entrez_not_found += 1
            final_results[cur_entrez] = 'No_Entrez_Found'
    print('number of genes not found: %d number of entrez not found: %d' %
          (num_gene_not_found, num_entrez_not_found))
    f_name = root_input + 'final_entrez_gene_map.pkl'
    pickle.dump(final_results, open(f_name, 'wb'))
    return final_results


def run_test(root_input, root_output):
    import scipy.io as sio
    f_name = root_input + 'original_data.pkl'
    original_data = pickle.load(open(f_name))
    f_name = 'results_bc_overlap_lasso_%02d.mat' % 0
    dataset = sio.loadmat(root_output + f_name)
    processed_data = dataset['data'][0][0]['XX']
    f_name = root_input + 'original_data.pkl'
    original_data = pickle.load(open(f_name))
    xx = np.mean(original_data['data_flag'], axis=1)
    print(len([_ for _ in xx if _ < 1e-3]))
    pass


def get_query(root_input):
    # install mygene
    import mygene

    mg = mygene.MyGeneInfo()
    f_name = root_input + 'gene_set.pkl'
    gene_set = pickle.load(open(f_name))
    gene_set = list(gene_set)
    out = mg.querymany(gene_set, scopes='symbol', fields='entrezgene',
                       species='human')
    pickle.dump(out, open(root_input + 'entrez_list.pkl', 'wb'))
    print(out)

    import scipy.io as sio

    f_name = 'entrez.mat'
    dataset = sio.loadmat(root_input + f_name)['entrez']
    entrez_list = [_[0] for _ in dataset]
    f_name = open(root_input + 'entrez.txt', 'a')
    print(f_name)
    for _ in entrez_list:
        f_name.write('%d\n' % _)
    f_name.close()
    exit()

    data = pickle.load(open(root_input + 'entrez_list.pkl'))
    f_open = open(root_input + 'gene_name_entrez_list.txt', 'a')
    index = 0
    for dat_ind, item in enumerate(data):
        if 'query' in item and 'entrezgene' in item:
            print('data_ind: %d gene name: %s entrezgene: %s' % (
                index, item[u'query'], item[u'entrezgene']))
            index += 1
            f_open.write('%s %s\n' % (item[u'query'], item[u'entrezgene']))
        else:
            print('data_ind: ---- gene name: %s' % (item[u'query']))

    f_open.close()


def test_03(root_input):
    import csv

    entrez_ids = dict()
    f_name = root_input + 'entrez.txt'
    with open(f_name) as csvfile:
        line_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row_ind, row in enumerate(line_reader):
            entrez_ids[int(row[0])] = ''
    entrez_gene_names = dict()
    f_name = root_input + 'entrez_gene_name.tab'
    with open(f_name) as csvfile:
        line_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row_ind, row in enumerate(line_reader):
            if row_ind >= 1:
                if str.find(row[-1], ',') != -1:
                    for entrez_id in str.split(row[-1], ','):
                        gene_names = str.split(row[-2], ' ')
                        gene_names = [_ for _ in gene_names if len(_) > 0]
                        if int(entrez_id) not in entrez_gene_names:
                            entrez_gene_names[int(entrez_id)] = []
                        entrez_gene_names[int(entrez_id)].extend(gene_names)
                else:
                    entrez_id = int(row[-1])
                    gene_names = str.split(row[-2], ' ')
                    gene_names = [_ for _ in gene_names if len(_) > 0]
                    if int(entrez_id) not in entrez_gene_names:
                        entrez_gene_names[int(entrez_id)] = []
                    entrez_gene_names[int(entrez_id)].extend(gene_names)
    index = 0
    for item in entrez_ids:
        if item not in entrez_gene_names:
            print(item)
            index += 1
    pass


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
    print(nx.number_connected_components(subgraph))
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
        print(method, found_set[method])
    data['found_related_genes'] = found_set
    return data


def summary_data(root_input):
    method_list = ['re_path_re_lasso', 're_path_re_overlap',
                   're_edge_re_lasso', 're_edge_re_overlap']
    summarized_data = dict()
    for folding_i in range(0, 20):
        print('processing folding_%02d' % folding_i)
        summarized_data[folding_i] = dict()
        data = get_single_data(folding_i, root_input)
        summarized_data[folding_i]['data_entrez'] = data['data_entrez']
        summarized_data[folding_i]['cancer_related_genes'] = \
            data['cancer_related_genes']
        summarized_data[folding_i]['found_related_genes'] = \
            data['found_related_genes']
        for method in method_list:
            summarized_data[folding_i][method] = dict()
        for method in method_list:
            all_involved_genes = set()
            bacc = [data[method][fold_i]['perf'][0][0]
                    for fold_i in range(5)]
            summarized_data[folding_i][method]['bacc'] = bacc
            auc, nonzeros_list = [], []
            for fold_i in range(5):
                kidx = data[method][fold_i]['kidx']
                summarized_data[folding_i][method]['kidx_%d' % fold_i] = kidx
                all_related_entrez = [data['data_entrez'][_] for _ in kidx]
                all_involved_genes = all_involved_genes.union(
                    set([data['cancer_related_genes'][_]
                         for _ in all_related_entrez
                         if _ in data['cancer_related_genes']]))
                best_lambda = data[method][fold_i]['lstar']
                ii = list(data[method][fold_i]['lambdas']).index(best_lambda)
                ws = data[method][fold_i]['oWs'][:, ii]
                nonzeros = np.nonzero(ws[1:])[0]
                summarized_data[folding_i][method]['ws_%d' % fold_i] = ws[1:]
                summarized_data[folding_i][method]['ws_nonzeros_%d' % fold_i] = \
                    nonzeros
                auc.append(data[method][fold_i]['auc'][0][0][ii])
                nonzeros_list.append(len(nonzeros))
            summarized_data[folding_i][method]['auc'] = auc
            summarized_data[folding_i][method]['num_nonzeros'] = nonzeros_list
            print(all_involved_genes)
    f_name = root_input + 'overlap_data_summarized.pkl'
    pickle.dump(summarized_data, open(f_name, 'wb'))


def main():
    summary_data(root_input='data/')


if __name__ == '__main__':
    main()
