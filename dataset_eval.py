import os
import numpy as np
import pandas as pd
import read_results
from collections import Counter
from plotter import save_pdf
import matplotlib.pyplot as plt


noise_params = ['00', '10', '20', '30', '40', '50']
output_path = os.path.dirname(__file__) + '/Graphs/'
input_path = os.path.dirname(__file__) + '/Results/'
markers = ['o-', 's-', '^-', 'd-', '*-', 'X-', 'D-', 'P-', '8-']


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
    k_neighbors = [str(k) for k in range(2, 8)]
    path_data = input_path + gen + "/" + dataset + "/" + dataset + "_" + technique + "_Test_Noise_"

    neigh_noise = pd.DataFrame(columns=list(map(str, range(2,8))))

    print(">>>>>> " + technique + " <<<<<<<")

    #Read the csv according to technique
    for noise in noise_params:
        data = pd.read_csv(path_data+noise+".csv")
        data.drop(data.columns[0], axis=1, inplace=True)

        data_comb = dict((str(k), []) for k in k_neighbors)

        for row in data.values:
            neighbors = Counter(row[:7])
            freq_neighbors = neighbors.values()

            if 2 in freq_neighbors:
                data_comb["2"].append(list(row))
            elif 3 in freq_neighbors:
                data_comb["3"].append(list(row))
            elif 4 in freq_neighbors:
                data_comb["4"].append(list(row))
            elif 5 in freq_neighbors:
                data_comb["5"].append(list(row))
            elif 6 in freq_neighbors:
                data_comb["6"].append(list(row))
            else:
                data_comb["7"].append(list(row))
        
        dataframes = dict.fromkeys(k_neighbors)
        for k in k_neighbors:
            dataframes[k] = pd.DataFrame(data_comb[k], columns=data.columns)
            neighbors_match = dataframes[k][(dataframes[k]["Target"] == dataframes[k]["Predictions"])].reset_index(drop=True)

            if len(dataframes[k]) == 0:
                percentage_neigh = None
            else:
                percentage_neigh = (len(neighbors_match)/len(dataframes[k]))*100
            
            neigh_noise.set_value(str(noise), str(k), percentage_neigh)
    
    print(neigh_noise)


def neighbors_techniques(gen, dataset, techniques, k):
    noise_df = dict((noise, pd.DataFrame()) for noise in noise_params)
    for technique in techniques:
        path_data = input_path + gen + "/" + dataset + "/" + dataset + "_" + technique + "_Test_Noise_"

        #Read the csv according to technique
        for noise in noise_params:
            data = pd.read_csv(path_data+noise+".csv")
            data.drop(data.columns[0], axis=1, inplace=True)

            data_comb = []
            for row in data.values:
                neighbors = Counter(row[:7])
                freq_neighbors = neighbors.values()
                if k in freq_neighbors:
                    data_comb.append(list(row))

            data = pd.DataFrame(data_comb, columns=data.columns)
            if noise_df[noise].empty:
                #Change the name of predictions
                data.rename(columns={"Predictions": "Predictions_"+str(technique)}, inplace=True)
                noise_df[noise] = data
            else:
                noise_df[noise]["Predictions_"+str(technique)] = data["Predictions"]

    accuracy_noise = pd.DataFrame(index = ["Errors"], columns=range(0,60,10))

    for noise in noise_params:
        noise_df[noise] = noise_df[noise][(noise_df[noise]["Target"] == noise_df[noise]["K1"])].reset_index(drop=True)
        print("Size: ", noise_df[noise].size)
        
        neighbors_match_OLA = noise_df[noise][(noise_df[noise]["Target"] == noise_df[noise]["Predictions_OLA"])].reset_index(drop=True)
        neighbors_match_LCA = noise_df[noise][(noise_df[noise]["Target"] == noise_df[noise]["Predictions_LCA"])].reset_index(drop=True)

        print("OLA - {}, LCA - {}".format(neighbors_match_OLA.size, neighbors_match_LCA.size))

        # accuracy_noise.set_value("Errors", int(noise), (neighbors_match.size/noise_df[noise].size)*100)
        # accuracy_noise.set_value("LCA", noise, (neighbors_match_LCA.size/noise_df[noise].size)*100)
    
    # accuracy_noise.T.plot(kind='line')
    # plt.xlabel("Noise level")
    # plt.ylabel("Errors %")
    # plt.xticks(rotation='horizontal')
    # plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=len(accuracy_noise.index))
    # plt.grid()
    # plt.show()
    
    return accuracy_noise        


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

def plot_df(df, gen, dataset, technique):
    markers = ['P', 'v', 's', '*', '.', 'X', 'o']

    ax = df.plot(kind='line')
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(markers[i])

    plt.xlabel("# Neighbors equals")
    plt.ylabel("Accuracy")
    plt.yticks(list(range(0,110, 10)))
    plt.xticks(rotation='horizontal')
    ax.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=len(df.index))
    plt.grid()
    save_pdf(plt, output_path, dataset + "_" + technique + "_Neighbors")


if __name__ == "__main__":
    datasets    = ['Kyoto2008']
    gen_methods = ['SGH']
    techniques = ['OLA', 'LCA']
    roc_dataset = dict.fromkeys(datasets, None)
    roc = list(range(2,8))

    df = pd.DataFrame(index=roc, columns=noise_params)
    for gen in gen_methods:
        for dataset in datasets:
            # for technique in techniques:
                # for k in roc:
                    # print(">>>>>>> {} <<<<<<<".format(k))
            # acc_df = neighbors_techniques(gen, dataset, techniques, 7)
            # print(acc_df)
                    # break
                #     df.loc[k] = acc_df.loc[str(technique)]

                # plot_df(df, gen, dataset, technique)

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