import os
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from Orange.evaluation import compute_CD, graph_ranks

from matplotlib.font_manager import FontProperties

from ensemble_utils import save_pdf

from read_results import read_mean_results, read_accuracies
from nonparametric_tests import friedman_test

import numpy as np

root = "../"
noise_params = ['00', '10', '20', '30', '40', '50']
# Plot configuration
markers = ['o', 's', '^', 'd', '*', 'X', 'D', 'P', '8', 'v', '.', 'x', '+', '*']
colors = ['rosybrown', 'indianred', 'firebrick', 'darkred', 'salmon', 'red', 'tomato', 'coral', 'orangered', 'sienna', 'chocolate', 'saddlebrown', 'sandybrown', 'peru', 'blue']

# Map name column
gen_map = {"BaggingClassifier": "Bagging", "AdaBoostClassifier": "AdaBoost", "SGH": "SGH"}

def plot__dataframes(dataframe1, dataframe2, dataset, classe):
    def autolabel(rects, col):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.005 * height,'%d' % int(height), ha='center', va='bottom', color=col, fontsize=8)

    output_path = os.path.dirname(__file__) + '/Graphs/' + dataset + "/Classes/"
    ind = dataframe1.index.astype('int')
    width = 0.35

    fig, axes = plt.subplots(ncols=1, nrows=len(noise_params), sharex=True)
    fig.text(0.55, 0.01, 'Neighbors with same label', ha='center')
    fig.text(-0.01, 0.55, 'Correctly Labeled', va='center', rotation='vertical')

    for i, ax in enumerate(axes.flatten()):
        OLA = ax.bar(ind, dataframe1.loc[:, noise_params[i]], width, color='b')
        LCA = ax.bar(ind + width, dataframe2.loc[:, noise_params[i]], width, color='r')
        ax.set_title('Noise {}%'.format(noise_params[i]), fontsize=9)
        autolabel(OLA, 'blue')
        autolabel(LCA, 'red')

    fig.legend((OLA, LCA), ('OLA', 'LCA'), loc='upper left', ncol=2)
    plt.tight_layout()
    save_pdf(plt, output_path, dataset + "_class_" + str(classe))


def plot__confusion_matrix(cm, classes, dataset, gen_method, dynamic_method, noise_level, title=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    output_path = root + '/Graphs/' + "/confusion_matrix/" + dataset + "/" + gen_method + "/"
    cmap = plt.get_cmap('Blues')
    sizeClasses = len(classes)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if classes is not None:
        classes = list(range(1, sizeClasses + 1))

        tick_marks = np.arange(sizeClasses)
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if sizeClasses > 5:
            if i == j:
                plt.text(j, i, "{:0.3f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.text(j, i, "{:0.3f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    save_pdf(plt, output_path, dynamic_method + "_" + noise_level)


def plot__single_technique(gen_methods, datasets, imb_method, metrics, techniques):
    for dataset in datasets:
        for metric in metrics:
            # Define output path
            output_path = root + '/Graphs/' + 'Single_technique/' + dataset + "/" + metric + "/"
            for technique in techniques:
                dataframe_list = []
                for gen in gen_methods:
                    # Get mean and standard deviation for each 
                    mean, std = read_mean_results(gen, dataset, imb_method, noise_params, metric, [technique])
                    # Rename the techniques that have been evaluated to add generation method information
                    mean.columns = [technique + "-" + gen_map[gen]]
                    # Append the information in a list of dataframes
                    dataframe_list.append(mean)
    
                # Techniques evaluation concatenated
                mean_techn = pd.concat(dataframe_list, axis=1)
                # Min value in each evaluation
                min_values = mean_techn.min(axis=1).min(axis=0)
    
                for i, column in enumerate(mean_techn):
                    if i < 4:
                        plt.plot(noise_params, mean_techn[column], colors[i], label=column, marker=markers[i],
                                     markersize=3, markerfacecolor=(1, 1, 1), linewidth=1)
                    else:
                        plt.plot(noise_params, mean_techn[column], colors[i], label=column, marker=markers[i - 4],
                                     markersize=3, linewidth=1)
        
                # Plot configuration
                legend_font = FontProperties()
                legend_font.set_size(8)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=False, shadow=False, prop=legend_font)
                plt.ylim(min_values, 101)
                plt.ylabel(metric)
                plt.xlabel('Noise per rate (%)')
                plt.title(dataset + ": " + metric)
                plt.grid()
                save_pdf(plt, output_path, technique + "_" + metric)


def plot__results(datasets, gen_methods, techniques, metrics, imb_method):
    output_path = root + '/Graphs/' + metric + '/' + gen + '/'

    mean_accuracies, std_acc = read_mean_accuracies_and_standard(datasets, gen_methods, techniques, metrics, imb_method, noise_params)

    for i, column in enumerate(mean_accuracies):
        plt.plot(noise_params, mean_accuracies[column], colors[i], label=column, marker=markers[i], markersize=3, linewidth=1)
    
    legend_font = FontProperties()
    legend_font.set_size(8)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=False, shadow=False, prop=legend_font)
    plt.ylim(0, 100)
    plt.ylabel(metric)
    plt.xlabel('Noise per rate (%)')
    save_pdf(plt, output_path, imb_method + "_" + dataset + "_" + metric)


def friedman_acc_test(datasets, gen_methods, metrics):
    dataset_folds = {}
    friedmanT = dict.fromkeys(noise_params, [])

    def plot__nemenyi(gen, ranks, techniques, noise):
        cd = compute_CD(ranks, 30, alpha="0.05")
        graph_ranks(ranks, techniques, cd=cd, width=len(techniques), textspace=1.5)
        save_pdf(plt, root + '/Graphs/' + '/Nemenyi/' + gen + '/', 'Nemenyi_' + noise)

    # Get mean and standard deviation
    # In Friedman Test, we're evaluating for Accuracy metric
    mean_accuracies, std_acc = read_mean_accuracies_and_standard(datasets, gen_methods, ['Accuracy'], metrics, imb_method, noise_params)
    dictionary = mean_accuracies.to_dict(orient='index')
    
    for k in dictionary:
        dataset_folds[k] = [dictionary[k][column_name] for column_name in mean_accuracies.columns]

    # Friedman Tests
    for key in friedmanT:
        friedmanT[key].append(dataset_folds[key])

    for key, value in friedmanT.items():
        value = list(zip(*value))
        Fvalue, pvalue, ranks, pivots = friedman_test(*value)
        plot_nemenyi(gen, ranks, techniques, key)

if __name__ == '__main__':
    # datasets = ['HH103', 'HH124', 'HH129', 'Kyoto2008', 'Kyoto2009Spring']
    datasets = ['HH124', 'Kyoto2008']
    gen_methods = ["BaggingClassifier", "AdaBoostClassifier", "SGH"]
    metrics = ['MultiLabel-Fmeasure', 'Gmean', 'Accuracy', 'Precision', 'Recall', 'F1']

    oracle = ['Oracle']
    baseline = ['RandomForestClassifier']
    techniques_dcs = ['OLA', 'LCA', 'Rank', 'MCB']
    techniques_des = ['KNORAU', 'KNORAE', 'DESKNN', 'DESP', 'DESMI', 'DESClustering', 'METADES', 'KNOP']

    techniques = baseline + techniques_dcs + techniques_des + oracle
    imb_method = "imbalanced"

    plot__single_technique(gen_methods, datasets, imb_method, metrics, techniques)
    # plot_free_noise(gen, data_type, datasets, metric, techniques)
