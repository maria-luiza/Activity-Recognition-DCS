import os
import numpy as np
import pandas as pd
import read_results
import itertools
from plotter import save_pdf
import matplotlib.pyplot as plt
import seaborn as sns


noise_params = ['00', '10', '20', '30', '40', '50']
output_path = os.path.dirname(__file__) + '/Graphs/'
markers = ['o-', 's-', '^-', 'd-', '*-', 'X-', 'D-', 'P-', '8-']

def plot_contourn():
    pass

def plot_roc(dict_data, datasets):
    df = pd.DataFrame.from_dict(dict_data, orient='columns')

    df.plot(kind='line', grid=True, style=markers)

    plt.ylim(0, 100)
    plt.ylabel('Match (%)')
    plt.xlabel('Noise per rate (%)')
    save_pdf(plt, output_path, 'ROC_datasets')

def plot_heatmap(dataset, df):

    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(dataset)

    for ax, noise in zip(axes.flatten(), df):
        heat = df[noise]
        ax.set_title(str(noise)+"%")
        sns.heatmap(heat, ax=ax, linewidths=0.5, fmt="d")

    save_pdf(plt, output_path+dataset+'/', dataset + "_" + str(noise))

def labels_changes_roc(gen, dataset, df1):
    labels = set().union(*[df1[col].unique() for col in df1.columns])
    mean_labels = dict.fromkeys(noise_params[1:], None)

    for noise in noise_params[1:]:
        dict_labels = {k: {t: 0 for t in labels} for k in labels}
        # Read the noisy dataset
        df2 = read_results.read_accuracies(gen, dataset, dataset, noise)

        for (_, s1 ), ( _, s2 ) in itertools.zip_longest( df1.iterrows(), df2.iterrows() ) :
            s1_l, s2_l = s1.tolist(), s2.tolist()

            for v1, v2 in zip(s1_l, s2_l):
                if v1 != v2:
                    dict_labels[v1][v2] += 1
        
        mean_labels[noise] = pd.DataFrame.from_dict(dict_labels, orient='columns')
    
    plot_heatmap(dataset, mean_labels)


def read_labels_roc(gen, dataset, std_result):
    # The free noise is the standard result
    mean_roc = dict.fromkeys(noise_params, None)

    for noise in noise_params:
        df = read_results.read_accuracies(gen, dataset, dataset, noise)
        # Check differences between the noisy environment and free noise
        df.where(df.values==std_result.values, inplace=True)
        # Replace NaN values for False. Otherwise, True
        df = df.notnull()
        # Calculate the mean for each row
        df['mean'] = df.mean(axis=1)

        # Mean between each roc
        mean_roc[noise] = df['mean'].mean(axis=0)*100
    return mean_roc


if __name__ == "__main__":
    datasets    = ['HH103', 'HH124', 'HH129', 'Kyoto2008', 'Kyoto2009Spring']
    gen_methods = ['SGH']
    roc_dataset = dict.fromkeys(datasets, None)

    for gen in gen_methods:
        for dataset in datasets:
            # # Get the matches on RoC per noise
            std_result = read_results.read_accuracies(gen, dataset, dataset, '00')
            # roc_dataset[dataset] = read_labels_roc(gen, dataset, std_result)
            # # Evaluate the changes in labels based on noise level
            labels_changes_roc(gen, dataset, std_result)
    # print(roc_dataset)
    # plot_roc(roc_dataset, datasets)