import os
import re
import docx
import shlex
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import cm
from plotter import *
from read_results import *

single_classifiers = ['SVM', 'Random Forest', 'KNN', 'Decision Tree']
datasets = ['HH103', 'HH105', 'HH110', 'HH124', 'HH125', 'HH126', 'HH129', 'Kyoto2008', 'Kyoto2009Spring', 'Tulum2009']
noise_params = ['00', '10', '20', '30', '40', '50']
root = os.path.dirname(__file__)

def load_folds(file_name):
    file_path = os.path.dirname(__file__) + '/folds/' + file_name
    with open(file_path, 'rb') as file:
        _fold = pickle.load(file)
        unique_activities_list = pickle.load(file)
        file.close()
    return _fold, unique_activities_list

def load_dataset(dataset_name):
    folds, activities_list = load_folds(dataset_name)
    labels_dict = {act: i for i, act in enumerate(activities_list)}
    activities_list = [labels_dict.get(x) for x in activities_list]
    return folds, activities_list, labels_dict


def get_text(filename):
    doc = docx.Document(filename)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return full_text


def string_to_confusion_matrix(conf_matrix):
    conf_matrix = conf_matrix.split('] [')
    for i in range(len(conf_matrix)):
        conf_matrix[i] = conf_matrix[i].replace('[', '')
        conf_matrix[i] = conf_matrix[i].replace(']', '')
        conf_matrix[i] = ','.join(shlex.split(conf_matrix[i]))
        conf_matrix[i] = conf_matrix[i].split(',')
        conf_matrix[i] = [int(n) for n in conf_matrix[i]]
    conf_matrix = np.array(conf_matrix)
    return conf_matrix


def read_single_classifiers_results():
    for dataset in datasets:
        for single_classifier_name in single_classifiers:
            file = single_classifier_name + '.docx'
            noise = fold = labels = None

            if os.path.exists(root + '/SingleClassifierResults/' + dataset + '/' + file):
                text = get_text(root + '/SingleClassifierResults/' + dataset + '/' + file)
                for line in text:
                    if '======== Noise Parameter -->' in line:
                        results = []
                        fold = 0
                        noise = re.search(pattern='[0-9]+', string=line).group()
                    if line.strip().startswith('General Accuracy'):
                        fold += 1
                        if fold < 6:
                            accuracy = line.split(' : ')[1]
                            idx = text.index(line) + 6
                            conf_matrix = ''
                            while ']]' not in line:  # Read the entire confusion matrix
                                line = text[idx]
                                conf_matrix += line
                                idx += 1
                            confusion_matrix = string_to_confusion_matrix(conf_matrix)
                            labels = [str(x) for x in range(len(confusion_matrix[0]) - 1)]
                            accuracy_by_class = {}
                            for label, acc in zip(labels, confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)):
                                accuracy_by_class[label] = acc
                            results.append(('Fold ' + str(fold), accuracy_by_class, accuracy, None))
                    if line.strip().startswith('Training Elapsed Time'):
                        save_results_df(build_results_df(results, labels), dataset, noise, single_classifier_name)

def build_results_df(metrics, results, labels):
    results_df  =  pd.DataFrame(columns = metrics, \
                                index = ['Fold ' + str(f) for f in range(1,6)] + ['Mean'])

    acc_class   = pd.DataFrame(columns = labels, \
                                index = ['Fold ' + str(f) for f in range(1,6)] + ['Mean'])

    for metric, functions in zip(metrics, results):
        functions = list(functions)
        dicts = functions.pop(2)

        results_df.loc[functions[0]]   = functions[1:]

        for key in dicts.keys():
            acc_class.loc[functions[0], key] = dicts[key]

    acc_class.loc['Mean']   = acc_class.mean(axis = 0)
    acc_class               = acc_class.fillna(0)
    results_df.loc['Mean']  = results_df.mean(axis = 0)

    return acc_class, results_df.reindex(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean'])

def save_results_df(accuracy_df, dataset, gen_method, noise, technique):
    file_path = root + '/Results/' + gen_method + '/' + dataset + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    noise = "%02d" % int(noise)
    noise = noise[1] + noise[0]
    accuracy_df.to_csv(file_path + technique + '_Noise_' + noise + '.csv', sep = ',')

def plot_histogram(dataset, noise, labels, values, tech):
    output_path = os.path.dirname(__file__) + '/Data/'+dataset+'/'
    indexes = np.arange(len(labels))
    width   = 0.5
    colors  = cm.Spectral(values / max(values))

    plot        = plt.scatter(values, values, c = values, cmap = 'Spectral')
    plt.clf()
    plt.colorbar(plot)
    
    _prop       = plt.bar(indexes, values, width, color = colors)
    plt.xticks(indexes)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height,
                    '%.2f' % height,
                    ha ='center', va = 'bottom')

    autolabel(_prop)
    
    plt.title(dataset+"__"+tech+"__noise_"+str(noise)+"0%")
    plt.xlabel("Labels")
    plt.ylabel("Proportion %")

    figure = plt.gcf() # get current figure
    figure.set_size_inches(14, 8)
    # plt.show()
    save_pdf(plt, output_path, dataset+"__"+tech+"_noise_"+str(noise)+"0")

if __name__ == '__main__':
    # datasets = ['HH103', 'HH124', 'HH129', 'Kyoto2008', 'Kyoto2009Spring']
    # datasets = ['HH124', 'HH129', 'Kyoto2008', 'Kyoto2009Spring']
    datasets = ["HH103"]

    for dataset in datasets:
        # Extract folds, list of activities and labels from each dataset
        # folds, activities_list, labels_dict = load_dataset(dataset)
        __labels = {}
        
        for noise in range(0,6):
            # for f, fold in enumerate(folds):
            #     args                = {'X_train': np.array(fold.xTrain)}
            #     y_train             = np.array(fold.yTrains[noise])
            #     # Brew requires numeric class labels
            #     args['y_train']     = np.array([labels_dict.get(x) for x in y_train])
            #     # args['X_test']      = np.array(fold.xTest)
            #     y_test              = np.array(fold.yTest)
            #     # Brew requires numeric class labels
            #     args['y_test']      = np.array([labels_dict.get(x) for x in y_test])
            #     # args['fold_name']   = 'Fold ' + str(f + 1)
            #     __labels[noise] = np.concatenate((args['y_train'], args['y_test']), axis = 0)
        
            # _count_labels           = Counter(__labels[noise])
            # labels, values_total    = zip(*sorted(_count_labels.items()))
            # total_counter           = sum(values_total)
            # values                  = []

            # for i, key in enumerate(labels):
            #     values.append(round(100*values_total[i]/total_counter, 2))

            # plot_histogram(dataset, noise, labels, np.array(values), "")

            # for balan in ["SMOTEENN", "RandomOverSampler", "SMOTE"]:
            for techn in ["Oracle"]:
                df = read_accuracies(dataset, techn+"_by_class", str(noise)+"0")
                df = 100*df.iloc[-1:]
                plot_histogram(dataset, noise, df.columns, np.array(df.values.tolist()[0]), techn)
