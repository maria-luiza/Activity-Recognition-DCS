import os
import numpy as np
import pandas as pd
import read_results
import itertools
from plotter import save_pdf
import matplotlib.pyplot as plt


noise_params = ['00', '10', '20', '30', '40', '50']
output_path = os.path.dirname(__file__) + '/Graphs/'
input_path = os.path.dirname(__file__) + '/Results/'
markers = ['o-', 's-', '^-', 'd-', '*-', 'X-', 'D-', 'P-', '8-']


def combination_columns(columns, n):
    col_combinations = list(itertools.combinations(columns, n))
    return list(map(list, col_combinations))


def class_per_noise_technique(gen, dataset, technique):
    path_data = input_path + gen + "/" + dataset + "/" + dataset + "_" + technique + "_Test_Noise_"
    data_noises = pd.DataFrame(columns = noise_params)

    print(">>>>>> " + technique + " <<<<<<<")

    #Read the csv according to technique
    for noise in noise_params:
        data = pd.read_csv(path_data+noise+".csv")
        data.drop(data.columns[0], axis=1, inplace=True)
        
        class_count = data.groupby("Predictions").size().reset_index(name='Counts')

        class_match = data.groupby(["Predictions", "Target"]).size().reset_index(name = "Match")
        class_match = class_match[(class_match["Predictions"] == class_match["Target"])].reset_index(drop=True)
        
        percentage_class = (class_match["Match"]/class_count["Counts"])*100
        data_noises[noise] = percentage_class
    
    data_noises.T.plot(kind='bar')
    plt.xlabel("Noise level")
    plt.ylabel("Accuracy")
    plt.xticks(rotation='horizontal')
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=len(class_match.index))
    save_pdf(plt, output_path, dataset + "_" + technique)

def neighbor_per_noise(gen, dataset, technique):
    path_data = input_path + gen + "/" + dataset + "/" + dataset + "_" + technique + "_Test_Noise_"
    neigh_noise = pd.DataFrame(columns=list(map(str, range(2,8))))
    markers = ['P', 'v', 's', '*', '.', 'X', 'o']

    print(">>>>>> " + technique + " <<<<<<<")

    #Read the csv according to technique
    for noise in noise_params:
        data = pd.read_csv(path_data+noise+".csv")
        data.drop(data.columns[0], axis=1, inplace=True)

        print("Noise level {}".format(noise))

        for n in range(2, 8):
            # print("-------------------- {} --------------------".format(n))
            comb_cols = combination_columns(data.columns[:7], n)
            data_n = []
            for cols in comb_cols:
                df1 = data[cols]
                df2 = data[[col for col in data.columns[:7] if col not in cols]]

                df = data[np.where(df1.eq(df1[cols[0]], axis=0).all(axis=1), True, False)\
                         & np.where(df2.ne(df1[cols[0]], axis=0).all(axis=1), True, False)]

                data_n.append(df)
            
            neigh = pd.concat(data_n)
            neigh_match = neigh[(neigh["Predictions"] == neigh["Target"])].reset_index(drop=True)

            # print(len(neigh_match)/len(neigh)*100)
            neigh_noise.set_value(str(noise), str(n), len(neigh_match)/len(neigh)*100)

    ax = neigh_noise.T.plot(kind='line')
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(markers[i])

    plt.xlabel("# Neighbors equals")
    plt.ylabel("Accuracy")
    plt.yticks(list(range(0,110, 10)))
    plt.xticks(rotation='horizontal')
    ax.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=len(neigh_noise.index))
    plt.grid()
    save_pdf(plt, output_path, dataset + "_" + technique + "_Neighbors")
    # print('-----------------------------------------------------------------------------------------------------------------')


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
    datasets    = ['Kyoto2008']
    gen_methods = ['SGH']
    techniques = ['OLA', 'LCA']
    roc_dataset = dict.fromkeys(datasets, None)

    for gen in gen_methods:
        for dataset in datasets:
            for technique in techniques:
                # class_per_noise_technique(gen, dataset, technique)
                neighbor_per_noise(gen, dataset, technique)
            # # Get the matches on RoC per noise
            # std_result = read_results.read_accuracies(gen, dataset, dataset, '00')
            # roc_dataset[dataset] = read_labels_roc(gen, dataset, std_result)
            # # Evaluate the changes in labels based on noise level
            # labels_changes_roc(gen, dataset, std_result)
    # print(roc_dataset)
    # plot_roc(roc_dataset, datasets)