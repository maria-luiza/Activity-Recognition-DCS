import os
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from Orange.evaluation import compute_CD, graph_ranks

from matplotlib.font_manager import FontProperties

from utils import save_pdf

from read_results import read_mean_results, read_accuracies
from nonparametric_tests import friedman_test

import numpy as np

root = "../"
noise_params = ['00', '10', '20', '30', '40', '50']


def plot_dataframes(dataframe1, dataframe2, dataset, classe):
    output_path = os.path.dirname(
        __file__) + '/Graphs/' + dataset + "/Classes/"
    ind = dataframe1.index.astype('int')
    width = 0.35

    fig, axes = plt.subplots(ncols=1, nrows=len(noise_params), sharex=True)
    fig.text(0.55, 0.01, 'Neighbors with same label', ha='center')
    fig.text(-0.01, 0.55, 'Correctly Labeled',
             va='center', rotation='vertical')

    for i, ax in enumerate(axes.flatten()):
        OLA = ax.bar(ind, dataframe1.loc[:, noise_params[i]], width, color='b')
        LCA = ax.bar(
            ind + width, dataframe2.loc[:, noise_params[i]], width, color='r')
        ax.set_title('Noise {}%'.format(noise_params[i]), fontsize=9)

        def autolabel(rects, col):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.005 * height,
                        '%d' % int(height),
                        ha='center', va='bottom', color=col, fontsize=8)

        autolabel(OLA, 'blue')
        autolabel(LCA, 'red')

    fig.legend((OLA, LCA), ('OLA', 'LCA'), loc='upper left', ncol=2)
    plt.tight_layout()
    save_pdf(plt, output_path, dataset + "_class_" + str(classe))


def plot_confusion_matrix(cm, classes,
                          dataset,
                          gen_method,
                          dynamic_method,
                          noise_level,
                          normalize=False,
                          title=None,
                          cmap=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    output_path = root + '/Graphs/' + "/confusion_matrix/" + dataset + "/" + gen_method + "/"

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    sizeClasses = len(classes)

    if classes is not None:
        classes = list(range(1, sizeClasses + 1))

        tick_marks = np.arange(sizeClasses)
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    def plotText(normalize=True):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if sizeClasses > 5:
            if i == j:
                plotText(normalize)
        else:
            plotText(normalize)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    save_pdf(plt, output_path, dynamic_method + "_" + noise_level)


def plot_technique(gen_methods, dataset, imb_method, metric, techniques):
    noise_params = ['00', '10', '20', '30', '40', '50']
    output_path = os.path.dirname(__file__) + '/Graphs/' + dataset + "/"
    markers = ['o', 's', '^', 'd', '*', 'X', 'D', 'P', '8']
    colors = ['blue', 'lightcoral', 'seagreen', 'crimson',
              'maroon', 'orangered', 'tomato', 'sienna']

    df_l = []
    for gen in gen_methods:
        df, std = read_mean_results(
            gen, dataset, imb_method, noise_params, metric, techniques)

        df.columns = [techniques[0] + "_" + gen] + techniques[1:]
        df_l.append(df)

    mean_techn = pd.concat(df_l, axis=1)
    for i, column in enumerate(mean_techn):
        if i < 4:
            plt.plot(noise_params, mean_techn[column], colors[i], label=column, marker=markers[i],
                     markersize=3, markerfacecolor=(1, 1, 1), linewidth=1)
        else:
            plt.plot(noise_params, mean_techn[column], colors[i], label=column, marker=markers[i - 4],
                     markersize=3, linewidth=1)

    legend_font = FontProperties()
    legend_font.set_size(8)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=4, fancybox=False, shadow=False, prop=legend_font)

    plt.ylim(30, 101)
    plt.ylabel(metric)
    plt.xlabel('Noise per rate (%)')
    plt.title(dataset)
    plt.grid()
    save_pdf(plt, output_path, imb_method + "_" +
             '_'.join(techniques) + "_" + metric)


def plot_results(gen, dataset, imb_method, metric, techniques):
    output_path = root + '/Graphs/' + metric + '/' + gen + '/'

    # Plot configuration
    markers = ['o', 's', '^', 'd', '*', 'X', 'D', 'P', '8', 'v', '.']
    colors = ['0.4', '0.6', 'lightcoral', 'red', 'crimson', 'maroon', 'orangered', 'tomato', 'sienna', 'salmon',
              'coral']

    mean_accuracies, std_acc = read_mean_results(
        gen, dataset, imb_method, noise_params, metric, techniques)

    for i, column in enumerate(mean_accuracies):
        if i < 4:
            plt.plot(noise_params, mean_accuracies[column], colors[i], label=column, marker=markers[i],
                     markersize=3, markerfacecolor=(1, 1, 1), dashes=[2, 5, 3, 5], linewidth=1)
        else:
            plt.plot(noise_params, mean_accuracies[column], colors[i], label=column, marker=markers[i - 4],
                     markersize=3, linewidth=1)
    legend_font = FontProperties()
    legend_font.set_size(8)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=4, fancybox=False, shadow=False, prop=legend_font)

    plt.ylim(0, 100)
    plt.ylabel(metric)
    plt.xlabel('Noise per rate (%)')
    save_pdf(plt, output_path, imb_method + "_" + dataset + "_" + metric)


def plot_free_noise(gen, imb_method, datasets, metric, techniques):
    free_noise = ['00']
    means = []
    output_path = root + '/Graphs/' + '/Free_noise/'

    for dataset in datasets:
        mean_accuracies, _ = read_mean_results(gen, dataset, imb_method, free_noise, metric, techniques)
        mean_accuracies.index = [dataset]
        means.append(mean_accuracies)

    df_means = pd.concat(means)
    ax = df_means.plot(kind="bar", grid=False)

    ax.set_xlabel('Datasets')
    ax.set_ylabel(metric)
    ax.set_xticklabels(datasets, rotation=45)
    plt.ylim(min(df_means.apply(lambda x: min(x))) - 10, 100)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, ncol=6)
    save_pdf(plt, output_path, "Free_noise_" + metric)


def plot_nemenyi(gen, ranks, techniques, noise):
    cd = compute_CD(ranks, 30, alpha="0.05")
    graph_ranks(ranks, techniques, cd=cd, width=len(techniques), textspace=1.5)
    save_pdf(plt, root + '/Graphs/' + '/Nemenyi/' + gen + '/', 'Nemenyi_' + noise)


if __name__ == '__main__':
    datasets = ['HH103', 'HH124', 'HH129', 'Kyoto2008', 'Kyoto2009Spring']
    gen_methods = ["BaggingClassifier", "AdaBoostClassifier", "SGH"]
    metrics = ['MultiLabel-Fmeasure', 'Gmean', 'Accuracy', 'Precision', 'Recall', 'F1']

    baseline = ['RandomForestClassifier']
    techniques_dcs = ['OLA', 'LCA', 'Rank', 'MCB']
    techniques_des = ['KNORAU', 'KNORAE', 'DESKNN', 'DESP', 'DESMI', 'DESClustering']

    techniques = baseline + techniques_dcs + techniques_des

    data_type = "imbalanced"

    friedmanT = {'00': [],
                 '10': [],
                 '20': [],
                 '30': [],
                 '40': [],
                 '50': []}

    # General measurements
    for gen in gen_methods:
        print("Generation >>>> {} <<<<".format(gen))
        for metric in metrics:
            for dataset in datasets:
                plot_results(gen, dataset, data_type, metric, techniques)
                mean_accuracies, std_accuracies = read_mean_results(
                    gen,
                    dataset, data_type,
                    noise_params,
                    metric,
                    techniques
                )

                if metric == "Accuracy":
                    dictionary = mean_accuracies.to_dict(orient='index')
                    dataset_folds = {}
                    for k in dictionary:
                        dataset_folds[k] = [dictionary[k][column_name] for column_name in mean_accuracies.columns]

                    # Friedman Tests
                    for key in friedmanT:
                        friedmanT[key].append(dataset_folds[key])

                    for key, value in friedmanT.items():
                        value = list(zip(*value))
                        Fvalue, pvalue, ranks, pivots = friedman_test(*value)
                        plot_nemenyi(gen, ranks, techniques, key)

            plot_free_noise(gen, data_type, datasets, metric, techniques)
