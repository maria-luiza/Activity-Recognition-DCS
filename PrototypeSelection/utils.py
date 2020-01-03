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


def build_results_df(metrics, results, labels):
    results_df  =  pd.DataFrame(columns = metrics, \
                                index = ['Fold_' + str(f) for f in range(1,6)] + ['Mean'])

    acc_class   = pd.DataFrame(columns = labels, \
                                index = ['Fold_' + str(f) for f in range(1,6)] + ['Mean'])
    
    for functions in results:
        functions = list(functions)
        dicts = functions.pop(2)
        results_df.loc[functions[0]] = functions[1:]

        for key in dicts.keys():
            acc_class.loc[functions[0], key] = dicts[key]

    acc_class.loc['Mean']   = acc_class.mean(axis = 0)
    acc_class               = acc_class.fillna(0)
    results_df.loc['Mean']  = results_df.mean(axis = 0)

    return acc_class, results_df.reindex(['Fold_' + str(f) for f in range(1,6)] + ['Mean'])

def save_results_df(accuracy_df, dataset, imb_method, gen_method, noise, technique):
    file_path = root + '/Results/' + imb_method + '/' + gen_method + '/' + dataset + '/'
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
