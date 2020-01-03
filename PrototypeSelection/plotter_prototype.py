import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
from read_results import read_mean_results, read_accuracies
from nonparametric_tests import friedman_test
from scipy.stats import wilcoxon

noise_params = ['00', '10', '20', '30', '40', '50']
proto_legends = ["ENN", "CNN", "RENN", "AllKNN", "TomekLinks", "NeighbourhoodCleaningRule"]
metric = "Accuracy"


def save_pdf(plot, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    with PdfPages(path + name + '.pdf') as pdf:
        d = pdf.infodict()
        d['Title'] = 'Results'
        d['Author'] = u'Maria Luiza'
        pdf.savefig(bbox_inches="tight", dpi = 100)
        plot.close()


def datasetPrototypeSelection(dataset, generation, dynamicTechniques, prototypeSelections):
    """
        Input:
            dataset - CASAS dataset used in the evaluation
            generation - Generation method used to create the pool of the classifiers
            prototypeSelections - List of Prototype Selection techniques
    """

    df_mean = dict.fromkeys(prototypeSelections, None)
    df_std = dict.fromkeys(prototypeSelections, None)
    
    for protoTech in prototypeSelections:
        mean, std = read_mean_results(generation, dataset, protoTech, noise_params, metric, dynamicTechniques)
        # Save mean and Standard Deviation for each Prototype Selection Technique
        df_mean[protoTech] = mean
        df_std[protoTech] = std

    return df_mean, df_std


def plotterDatasetSelection(dicts, dataset, dynamic, prototypeSelections):

    output = root + "/Graphs/Datasets/"
    legend_font = FontProperties()
    legend_font.set_size(9)

    df = pd.DataFrame(columns = prototypeSelections, index = noise_params)
    for protoTech in prototypeSelections:
        df.loc[:, protoTech] = dicts[protoTech][dynamic]

    df.plot(
        kind = "line",
        ax = plt.figure().gca(),
        grid = True,
        lw=0.8,
        style = ['-o', '-s', '-^', '-d', '-*', '-X', '-D', '-P', '-8']
    )

    plt.ylabel(metric)
    plt.xlabel('Noise per rate (%)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=len(prototypeSelections), labels=proto_legends, prop=legend_font)
    save_pdf(plt, output, dataset + "_" + dynamic + "_" + metric)

def plotterPrototypeSelection(dicts, dataset, prototypeSelection):

    output = root + "/Graphs/"
    legend_font = FontProperties()
    legend_font.set_size(9)

    df = dicts[prototypeSelection]

    df.plot(
        kind = "line",
        ax = plt.figure().gca(),
        grid = True,
        lw=0.8,
        style = ['-o', '-s', '-^', '-d', '-*', '-X', '-D', '-P', '-8']
    )

    plt.ylabel(metric)
    plt.xlabel('Noise per rate (%)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=len(df.columns), prop=legend_font)
    save_pdf(plt, output + prototypeSelection + "/", dataset + "_" + prototypeSelection + "_" + metric)

if __name__ == '__main__':
    # Main
    root = os.path.dirname(__file__)
    gen_methods = ["SGH"]
    ds_methods = ["OLA", "LCA", "MCB", "Rank", "RandomForestClassifier"]
    proto_selections = ["EditedNearestNeighbours", "CondensedNearestNeighbour", "RepeatedEditedNearestNeighbours", "AllKNN", "TomekLinks", "NeighbourhoodCleaningRule"]
    datasets = ['HH103', 'HH124', 'HH129', 'Kyoto2008', 'Kyoto2009Spring']

    for dataset in datasets:
        print("Dataset >>> {} <<<".format(dataset))
        for gen_method in gen_methods:
            df_mean, df_std = datasetPrototypeSelection(dataset, gen_method, ds_methods, proto_selections)
            
            for ds_method in ds_methods:
                print("Dynamic Selection >>> {} <<<".format(ds_method))
                plotterDatasetSelection(df_mean, dataset, ds_method, proto_selections)
            
            print("\n")
            for proto_tech in proto_selections:
                print("Prototype Selection >>> {} <<<".format(proto_tech))
                plotterPrototypeSelection(df_mean, dataset, proto_tech)
        
        print("\n\n")