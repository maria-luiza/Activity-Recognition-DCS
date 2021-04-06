import os
import pandas as pd
import numpy as np

root = "../"
noise_params = ['00', '10', '20', '30', '40', '50']
input_path = root + '/Results/'


def read_mean_results(gen, dataset, imb_method, noise_params, metric, techniques, data_type):
    mean_accuracies = pd.DataFrame(0, columns=techniques, index=noise_params)
    std_dev_accuracies = pd.DataFrame(0, columns=techniques, index=noise_params)

    for noise in noise_params:
        for technic in techniques:
            acc = read_accuracies(gen, dataset, imb_method, technic, noise, data_type)

            if technic == "Random Forest" and metric != "Accuracy":
                mean_accuracies.loc[noise, technic] = round(acc.loc['Mean', metric + " Micro"] * 100, 2)
                std_dev_accuracies.loc[noise, technic] = round(np.std(acc[metric + ' Micro']) * 100, 2)

            else:
                mean_accuracies.loc[noise, technic] = round(acc.loc['Mean', metric] * 100, 2)
                std_dev_accuracies.loc[noise, technic] = round(np.std(acc[metric]) * 100, 2)

    mean_accuracies.index = noise_params
    return mean_accuracies, std_dev_accuracies


def read_accuracies(gen, dataset, imb_method, technic, noise, data_type):
    return pd.read_csv(input_path +
                       imb_method + '/' +
                       gen + '/' +
                       data_type + '/' +
                       dataset + '/' +
                       technic +
                       '_Noise_' +
                       noise + '.csv',
                       index_col=0, header=0)


def read_mean_accuracies_and_standard(dataset, gen, techniques, metric, imb_method, noise_params, data_type):
    mean_accuracies, std_acc = read_mean_results(gen, dataset, imb_method, noise_params, metric, techniques, data_type)

    return mean_accuracies, std_acc


def read_mean_results_per_noise(datasets, gen_method, imb_method, noise, techniques, data_type):
    mean_accuracies = pd.DataFrame(index=techniques, columns=datasets)
    std_dev_accuracies = pd.DataFrame(index=techniques, columns=datasets)
    for dataset in datasets:
        for technic in techniques:
            acc = read_accuracies(gen_method, dataset, imb_method, technic, noise, data_type)
            mean_accuracies.loc[technic, dataset] = acc.loc['Mean', 'General accuracy']
            std_dev_accuracies.loc[technic, dataset] = np.std(acc['General accuracy'][0: -1])

    mean_accuracies.index = techniques
    return mean_accuracies, std_dev_accuracies


def read_folds_accuracies(gen_method, dataset, imb_method, technic, noise, data_type):
    return read_accuracies(gen_method, dataset, imb_method, technic, noise, data_type)['General accuracy'][0: -1]


def read_folds_results_per_noise(gen_method, datasets, imb_method, techniques, noise, data_type):
    accuracies = pd.DataFrame(columns=techniques)
    temp = pd.DataFrame(columns=techniques)
    for dataset in datasets:
        for technic in techniques:
            acc = read_folds_accuracies(gen_method, dataset, imb_method, technic, noise, data_type)
            temp[technic] = acc
        accuracies = accuracies.append(temp)

    accuracies.index = range(len(accuracies))
    return accuracies
