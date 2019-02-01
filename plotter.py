import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Orange.evaluation import compute_CD, graph_ranks

from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

from read_results import read_mean_results, read_accuracies
from nonparametric_tests import friedman_test
from scipy.stats import wilcoxon

import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title = 'Confusion matrix',
                          cmap = None,
                          normalize = True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    output_path = os.path.dirname(__file__) + '/Graphs/' + title.split("__")[0] + "/"

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy = {:0.4f}; misclass = {:0.4f}'.format(accuracy, misclass))
    save_pdf(plt, output_path, "Confusion_Matrix__" + title)

def save_pdf(plot, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    with PdfPages(path + name + '.pdf') as pdf:
        d = pdf.infodict()
        d['Title'] = 'Results'
        d['Author'] = u'Maria Luiza'
        pdf.savefig(bbox_inches="tight", dpi = 100)
        plot.close()


def plot_technique(gen_methods, dataset, metric, technique):
    noise_params = ['00', '10', '20', '30', '40', '50']
    output_path = os.path.dirname(__file__) + '/Graphs/' + dataset + "/"
    markers = ['o', 's', '^', 'd', '*', 'X', 'D', 'P', '8']
    colors = ['blue', 'lightcoral', 'seagreen', 'crimson', 'maroon', 'orangered', 'tomato', 'sienna']

    df_l = []
    for gen in gen_methods:
        df = read_mean_results(gen, dataset, noise_params, metric, [technique])
        df.columns = [technique + "_" + gen] 
        df_l.append(df)
    
    mean_techn = pd.concat(df_l, axis=1)
    for i, column in enumerate(mean_techn):
        if i < 4:
            plt.plot(noise_params, mean_techn[column], colors[i], label=column, marker=markers[i],
                    markersize=3, markerfacecolor=(1, 1, 1), linewidth=1)
        else:
            plt.plot(noise_params, mean_techn[column], colors[i], label=column, marker=markers[i-4],
                    markersize=3, linewidth=1)
    
    legend_font = FontProperties()
    legend_font.set_size(8)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=False, shadow=False, prop=legend_font)

    plt.ylim(0, 100)
    plt.ylabel(metric)
    plt.xlabel('Noise per rate (%)')
    plt.title(dataset)

    plt.grid()
    save_pdf(plt, output_path, technique + "_" + metric)


def plot_results(gen, dataset, metric, techniques):
    noise_params = ['00', '10', '20', '30', '40', '50']
    output_path = os.path.dirname(__file__) + '/Graphs/' + gen + '/'
    markers = ['o', 's', '^', 'd', '*', 'X', 'D', 'P', '8']
    colors = ['0.4', '0.6', 'lightcoral', 'red', 'crimson', 'maroon', 'orangered', 'tomato', 'sienna']

    mean_accuracies = read_mean_results(gen, dataset, noise_params, metric, techniques)
    for i, column in enumerate(mean_accuracies):
        if i < 4:
            plt.plot(noise_params, mean_accuracies[column], colors[i], label=column, marker=markers[i],
                     markersize=3, markerfacecolor=(1, 1, 1), dashes=[2, 5, 3, 5], linewidth=1)
        else:
            plt.plot(noise_params, mean_accuracies[column], colors[i], label=column, marker=markers[i-4],
                     markersize=3, linewidth=1)
    legend_font = FontProperties()
    legend_font.set_size(8)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=False, shadow=False, prop=legend_font)

    plt.ylim(0, 100)
    plt.ylabel(metric)
    plt.xlabel('Noise per rate (%)')

    plt.grid()
    save_pdf(plt, output_path, dataset + "_" + metric)

def plot_free_noise(datasets, metric, techniques):
    noise_params    = ['00']
    means           = []
    output_path     = os.path.dirname(__file__) + '/Graphs/'

    for i, dataset in enumerate(datasets):
        mean_accuracies = read_mean_results(dataset, noise_params, metric, techniques)
        mean_accuracies.index = [dataset]
        means.append(mean_accuracies)
    
    df_means = pd.concat(means)
    ax = df_means.plot(kind = "bar", grid = True)
    
    ax.set_xlabel('Datasets')
    ax.set_ylabel(metric)
    ax.set_xticklabels(datasets, rotation = 45)
    plt.ylim(min(df_means.apply(lambda x: min(x))) - 10,100)

    plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.1), fancybox = True, ncol = 6)
    save_pdf(plt, output_path, "Free_noise_" + metric)

def plot_nemenyi(ranks, techniques, noise):
    cd = compute_CD(ranks, 45)
    graph_ranks(ranks, techniques, cd = cd, width = 6, textspace = 1.5)

    save_pdf(plt, os.path.dirname(__file__) + '/Graphs/', 'Nemenyi_' + noise)

if __name__ == '__main__':
    datasets    = ['HH103', 'HH124', 'HH129', 'Kyoto2008', 'Kyoto2009Spring']
    metrics     = ['Accuracy', 'Precision', 'Recall', 'F1']
    techniques  = ['Oracle', 'OLA', 'LCA', 'MLA', 'Rank', 'KNORAE', 'KNORAU', 'DESKNN', 'Random Forest']
    gen_methods = ['SGH', 'AdaBoostClassifier', 'BaggingClassifier']
    tableLatex  = open("tableLatex.txt", 'w')

    friedmanT  = {'00': [],
                  '10': [],
                  '20': [],
                  '30': [],
                  '40': [],
                  '50': []}
    
    for gen in gen_methods:
        for metric in metrics:
            for dataset in datasets:
                mean_accuracies = read_mean_results(gen, dataset, ['00', '10', '20', '30', '40', '50'], metric, techniques)
                tableLatex.write(mean_accuracies.to_latex())

                plot_results(gen, dataset, metric, techniques)
    
    for metric in metrics:
        for dataset in datasets:
            print("Dataset: ", dataset)
            plot_technique(gen_methods, dataset, metric, "Oracle")
    
        #     if metric == 'Accuracy':
        #         for key in friedmanT:
        #             mean = []
        #             for tech in techniques:
        #                 mean.append(mean_accuracies.loc[key, tech])

        #             friedmanT[key].append(mean)

        # for key, value in friedmanT.items():
        #     value = list(zip(*value))
        #     Fvalue, pvalue, ranks, pivots = friedman_test(value[0], value[1], value[2], value[3], value[4], value[5])
        #     plot_nemenyi(ranks, techniques, key)

        # plot_free_noise(datasets, metric, techniques)

    # sufix = ['_by_class_Noise_00', '_by_class_Noise_10', '_by_class_Noise_20',
    #          '_by_class_Noise_30', '_by_class_Noise_40', '_by_class_Noise_50']

    # for classes in sufix:
    #     for dataset in datasets:
    #         excel_names = ['Results/' + dataset + "/" + tech + classes + '.csv' for tech in techniques]
    #         excels      = [pd.read_csv(name, header = 0) for name in excel_names]
    #         excels      = [excel.loc[[len(excel)]] for excel in excels]
            # combined    = pd.concat(excels)
            # combined.to_csv('Graphs/Class/' + dataset + classes + '.csv', encoding = 'utf-8')