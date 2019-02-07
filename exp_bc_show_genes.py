import pickle
import numpy as np


def summarize_data(method_list, folding_list, num_iterations, root_output):
    sum_data = dict()
    cancer_related_genes = {
        4288: 'MKI67', 1026: 'CDKN1A', 472: 'ATM', 7033: 'TFF3', 2203: 'FBP1',
        7494: 'XBP1', 1824: 'DSC2', 1001: 'CDH3', 11200: 'CHEK2',
        7153: 'TOP2A', 672: 'BRCA1', 675: 'BRCA2', 580: 'BARD1', 9: 'NAT1',
        771: 'CA12', 367: 'AR', 7084: 'TK2', 5892: 'RAD51D', 2625: 'GATA3',
        7155: 'TOP2B', 896: 'CCND3', 894: 'CCND2', 10551: 'AGR2',
        3169: 'FOXA1', 2296: 'FOXC1'}
    for trial_i in folding_list:
        sum_data[trial_i] = dict()
        f_name = root_output + 'results_exp_bc_%02d_%02d.pkl' % \
                 (trial_i, num_iterations)
        data = pickle.load(open(f_name))
        for method in method_list:
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


def show_test(nonconvex_method_list, folding_list, max_epochs, root_input, root_output, latex_flag=True):
    sum_data = summarize_data(nonconvex_method_list,
                              folding_list, max_epochs, root_output)
    all_data = pickle.load(open(root_input + 'overlap_data_summarized.pkl'))
    for trial_i in sum_data:
        for method in nonconvex_method_list:
            all_data[trial_i]['re_%s' % method] = sum_data[trial_i][method]
        for method in ['re_%s' % _ for _ in nonconvex_method_list]:
            re = all_data[trial_i][method]['found_genes']
            all_data[trial_i]['found_related_genes'][method] = set(re)
    method_list = ['re_path_re_lasso', 're_path_re_overlap',
                   're_edge_re_lasso', 're_edge_re_overlap',
                   're_iht', 're_sto-iht', 're_graph-iht', 're_graph-sto-iht']
    all_involved_genes = {method: set() for method in method_list}
    for trial_i in sum_data:
        for method in nonconvex_method_list:
            all_data[trial_i]['re_%s' % method] = sum_data[trial_i][method]
        for method in ['re_%s' % _ for _ in nonconvex_method_list]:
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
        print('Edge-Lasso    Edge-Overlap  IHT           StoIHT        '
              'GraphIHT      GraphStoIHT')
        for folding_i in folding_list:
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
        best_bacc_dict = {_: 0 for _ in range(8)}
        best_auc_dict = {_: 0 for _ in range(8)}
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
                ['Folding ID',
                 '$\ell_1$-\\textsc{Pathway}',
                 '$\ell^1/\ell^2$-\\textsc{Pathway}',
                 '$\ell_1$-\\textsc{Edge}',
                 '$\ell^1/\ell^2$-\\textsc{Edge}',
                 '\\textsc{IHT}',
                 '\\textsc{StoIHT}',
                 '\\textsc{GraphIHT}',
                 '\\textsc{GraphStoIHT}'])),
            print('\\\\')
            print('\hline')

            for folding_i in folding_list:
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
                        row_list.append('%.3f$\pm$%.2f' % (x1, x2))
                if metric == 'bacc':
                    min_index = np.argmin(find_min)
                    original_string = row_list[int(min_index)]
                    mean_std = original_string.split('$\pm$')
                    row_list[int(min_index)] = '\\textbf{%s}$\pm$%s' % (
                        mean_std[0], mean_std[1])
                    best_bacc_dict[min_index - 1] += 1
                elif metric == 'auc':
                    max_index = np.argmax(find_max)
                    original_string = row_list[int(max_index)]
                    mean_std = original_string.split('$\pm$')
                    row_list[int(max_index)] = '\\textbf{%s}$\pm$%s' % (
                        mean_std[0], mean_std[1])
                    best_auc_dict[max_index - 1] += 1
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
                    row_list.append('%.3f$\pm$%.2f' % (x1, x2))
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
            print('\n\n\n\n')
        print('_' * 164)

        for ind, _ in enumerate(method_list):
            print(_, best_bacc_dict[ind], best_auc_dict[ind])
    found_genes = {method: set() for method in method_list}
    for folding_i in folding_list:
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


def show_detected_genes_2():
    import networkx as nx
    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = 8, 5.
    from matplotlib import rc
    from matplotlib.font_manager import FontProperties
    plt.rc('font', **{'size': 12, 'weight': 'bold'})
    rc('text', usetex=True)
    font0 = FontProperties()
    font0.set_weight('bold')
    all_subgraphs = pickle.load(open('results/kegg/generated_subgraphs.pkl'))
    method_list = ['rda-l1', 'da-iht', 'adagrad', 'da-gl', 'da-sgl', 'graph-da-iht']
    label_list = [r'$\displaystyle \ell_1$-\textsc{RDA}', r'\textsc{DA-IHT}', r'\textsc{AdaGrad}',
                  r'\textsc{DA-GL}', r'\textsc{DA-SGL}', r'\textsc{GraphDA}']
    fig, ax = plt.subplots(5, 6)
    for trial_i_ind, trial_i in enumerate(range(0, 5)):
        for method_ind, method in enumerate(method_list):
            print(trial_i, method)
            g = nx.Graph()
            detected_graph = all_subgraphs['s1'][method][trial_i]
            detected_nodes = list(detected_graph['nodes'])
            true_nodes = all_subgraphs['hsa05213']['nodes']
            intersect = list(set(true_nodes).intersection(detected_graph['nodes']))
            for edge_ind, edge in enumerate(detected_graph['edges']):
                g.add_edge(edge[0], edge[1], weight=detected_graph['weights'][edge_ind])
                for node in list(detected_graph['nodes']):
                    g.add_node(node)
                    for node in list(true_nodes):
                        g.add_node(node)
                        g = nx.minimum_spanning_tree(g)
                        print('method: %s pre: %.3f rec: %.3f fm: %.3f' %
                              (method, float(len(intersect)) / float(len(detected_nodes)),
                               float(len(intersect)) / float(len(true_nodes)),
                               float(2.0 * len(intersect)) / float(len(detected_nodes) + len(true_nodes))))
                        color_list = []
                        for node in detected_nodes:
                            if node in intersect:
                                color_list.append('r')
                            else:
                                color_list.append('b')
                                nx.draw_spring(g, ax=ax[trial_i_ind, method_ind], node_size=10, edge_color='black',
                                               edge_width=2.,
                                               font_size=4, node_edgecolor='black', node_facecolor='white',
                                               node_edgewidth=1., k=10.0, nodelist=detected_nodes,
                                               node_color=color_list)
                                ax[trial_i_ind, method_ind].axis('on')
                                ax[0, method_ind].set(title='%s' % label_list[method_ind])
                                plt.setp(ax[trial_i_ind, method_ind].get_xticklabels(), visible=False)
                                plt.setp(ax[trial_i_ind, method_ind].get_yticklabels(), visible=False)
                                ax[trial_i_ind, method_ind].tick_params(axis='both', which='both', length=0)
                                plt.subplots_adjust(wspace=0.0, hspace=0.0)
                                f_name = 'results/kegg/figs/hsa05213_%s_all_0_5.pdf' % 's1'
                                plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0.03, format='pdf')
                                plt.close()


def main():
    pass


if __name__ == '__main__':
    main()
