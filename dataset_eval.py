import os
import numpy as np
import pandas as pd
import read_results
from collections import Counter
from plotter import save_pdf, plot_dataframes, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


noise_params = ['00', '10', '20', '30', '40', '50']
output_path = os.path.dirname(__file__) + '/Graphs/'
input_path = os.path.dirname(__file__) + '/Results/'
markers = ['o', 's', '^', 'd', '*', 'X', 'D', 'P', '8']


def fold_class(dataset, gen, techniques, noise):
    k_neighbors = [str(k) for k in range(2, 8)]
    neigh_noise = pd.DataFrame(columns=list(map(str, range(2,8))))
    
    for fold in range(1,6):
        techn = pd.DataFrame(index=techniques, columns=list(map(str, range(2,8))))
        for technique in techniques:
            path_folds = input_path + gen + "/Folds/" + dataset + "/" + dataset + "_" + technique + "_Fold_"

            data = pd.read_csv(path_folds +str(fold)+ "_Noise_"+noise+".csv")
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
                neigh_noise.set_value(str(noise), str(k), len(neighbors_match))
        
            techn.loc[technique] = neigh_noise.values
        
        ax = techn.T.plot(kind='bar')
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        plt.title(noise)
        # plt.xlabel("Neighbors")
        # plt.ylabel("Correctly labeled")
        # plt.xticks(rotation='horizontal')
    
        plt.show()    


def class_per_noise_technique(gen, dataset, technique):
    path_data = input_path + gen + "/" + dataset + "/" + dataset + "_" + technique + "_Test_Noise_"
    data_noises = pd.DataFrame(columns = noise_params)

    print(">>>>>> " + technique + " <<<<<<<")

    #Read the csv according to technique
    for noise in noise_params:
        data = pd.read_csv(path_data+noise+".csv")
        data.drop(data.columns[0], axis=1, inplace=True)
        
        # Size of classes on Dataset
        class_count = data.groupby("Target").size().reset_index(name='Counts')

        # Filter dataframe based on correctly labeled
        class_match = data[data['Predictions'] == data['Target']]
        class_match = class_match.groupby("Target").size().reset_index(name="Match")
        
        percentage_class = (class_match["Match"]/class_count["Counts"])*100
        data_noises[noise] = percentage_class
    
    print(data_noises.T)
    data_noises.T.plot(kind='bar')
    plt.xlabel("Noise level")
    plt.ylabel("Accuracy")
    plt.xticks(rotation='horizontal')
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=len(class_match.index))
    # plt.show()
    save_pdf(plt, output_path, dataset + "_" + technique)

def neighbor_per_noise(gen, dataset, technique, classe):
    k_neighbors = [str(k) for k in range(2, 8)]
    neigh_noise = pd.DataFrame(columns=list(map(str, range(2,8))))
    path_data = input_path + gen + "/" + dataset + "/" + dataset + "_" + technique + "_Test_Noise_"

    #Read the csv according to technique
    for noise in noise_params:
        data = pd.read_csv(path_data+noise+".csv")
        data.drop(data.columns[0], axis=1, inplace=True)
        data = data[data['Target'] == classe]

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

        for k in k_neighbors:
            data_neighbors = pd.DataFrame(data_comb[k], columns=data.columns)
            neighbors_match = data_neighbors[(data_neighbors["Target"] == data_neighbors["Predictions"])].reset_index(drop=True)
            neigh_noise.set_value(str(noise), str(k), len(neighbors_match.index))

    return neigh_noise.T


def neighbors_k_techniques(gen, dataset, technique, k, classe):
    path_data = input_path + gen + "/" + dataset + "/" + dataset + "_" + technique + "_Test_Noise_"
    perc = []
    sizes = []
    #Read the csv according to technique
    for noise in noise_params:
        data = pd.read_csv(path_data+noise+".csv")
        data.drop(data.columns[0], axis=1, inplace=True)

        data = data[data['Target'] == classe]

        data_comb = []
        for row in data.values:
            neighbors = Counter(row[:7])
            freq_neighbors = neighbors.values()
            if k in freq_neighbors:
                minor = list(neighbors.keys())[1:]
                values = list(row) + [minor]
                data_comb.append(values)

        columns = np.append(data.columns, "Minor")
        data = pd.DataFrame(data_comb, columns=columns)

        minor_predictions = data[data.apply(lambda x: x['Predictions'] in x['Minor'], axis=1)]

        perc_minor = (len(minor_predictions)/len(data))*100

        sizes.append(len(data))
        perc.append(round(perc_minor,2))
    
    print("Neighbors {} - Size {} - Classe {} = {}".format(k, sizes, classe, perc))


def neighbors_techniques(gen, dataset, technique, classe):
    path_data = input_path + gen + "/" + dataset + "/" + dataset + "_" + technique + "_Test_Noise_"
    k_neighbors = [str(k) for k in range(2, 8)]

    #Read the csv according to technique
    for noise in noise_params:
        print(" >>>>>>>>> Noise {} <<<<<<<<<<<<<".format(noise))
        data = pd.read_csv(path_data+noise+".csv")
        data.drop(data.columns[0], axis=1, inplace=True)

        data = data[data['Target'] == classe]

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

        perc = []
        for k in k_neighbors:
            data_neighbors = pd.DataFrame(data_comb[k], columns=data.columns)
            hit_data = data_neighbors[data_neighbors['Target'] == data_neighbors['Predictions']]
            acc_tech = round((len(hit_data)/len(data))*100, 2)
            
            perc.append(acc_tech)

        print("Classe {} = {}".format(classe, perc))

def confusion_classes(gen, dataset, technique):
    path_data = input_path + gen + "/" + dataset + "/" + dataset + "_" + technique + "_Test_Noise_"

    print(">>>>>> " + technique + " <<<<<<<")

    #Read the csv according to technique
    for noise in noise_params:
        data = pd.read_csv(path_data+noise+".csv")
        data.drop(data.columns[0], axis=1, inplace=True)

        cnf = confusion_matrix(data[['Target']], data[['Predictions']])
        plot_confusion_matrix(cnf, list(range(0,5)), normalize=True, title=technique+"_noise_"+noise)
        save_pdf(plt, output_path + "Confusion/", dataset + "_" + technique + "_noise_" + noise)

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
    roc = list(range(2,8))

    df = pd.DataFrame(index=roc, columns=noise_params)
    for gen in gen_methods:
        for dataset in datasets:
            # for noise in noise_params:
            #     fold_class(dataset, gen, techniques, noise)
            # for class in range(0,5):
            #     tech = []
            for technique in techniques:
                    # class_per_noise_technique(gen, dataset, technique)
                    # tech.append(neighbor_per_noise(gen, dataset, technique, class))
                # confusion_classes(gen, dataset, technique)
                print(" >>> Technique << {} ".format(technique))
                # for k in range(5,7): # Apenas 5 e 6 vizinhos
                for classe in [1,4]:
                        # neighbors_k_techniques(gen, dataset, technique, k, classe)
                        # print("Classe {}".format(classe))
                    neighbors_techniques(gen, dataset, technique, classe)
                print("------------------------------------------------------------------")

                # plot_dataframes(tech[0], tech[1], dataset, classe) #OLA #LCA
