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


def dataframe_pred(dataframe, columns, scenario):
    # Number of Base classifiers
    cols_B = [col for col in columns if "B" in col]
    
    if not dataframe.empty:
        # Filter by predictions
        data_pred = dataframe[dataframe['Target'] == dataframe['Predictions']]

        if not data_pred.empty:
            # Filter by Base Classifiers
            df_out = data_pred.filter(like='B',axis=1)
            return df_out

    print("There's no target in {}.".format(scenario))
    out = pd.DataFrame(0, index=np.arange(len(cols_B)), columns=cols_B)

    return out


def get_perc(df1, df2):
    perc = 0
    if not df2.empty:
        perc = 100*len(df1)/len(df2)
    
    return perc


def competence_neighbors(path, dataset, gen, technique, ks):
    folder = input_path + gen + "/" + dataset + "/" + path + "/" +dataset + "_" + technique + "_" + path + "_Noise_"

    print("Technique: ", technique)

    for noise in noise_params:
        print("Noise: {}".format(noise))
        # Read the data
        data = pd.read_csv(folder+noise+".csv")
        data.drop(data.columns[0], axis=1, inplace=True)

        # Get classes
        classes = set(data['Target'])

        for classe in classes:
            print("Classe {}".format(classe))
            
            # Filter dataset based on class
            data_classe = data[data['Target'] == classe]
            # Reset indexes
            data_classe.reset_index(drop=True, inplace=True)
            
            # Get neighbors classes
            neighbors_df = data_classe.filter(like='K', axis=1)

            for k in ks:
                print("Neighbors: ", k)
                minors = [] # Relative Minority Class
                majorities = [] # Reltive Majority Class
                ind = [] # Index of nieghbors

                # Get indexes and values for each neighbor
                for index, row in enumerate(neighbors_df.values):
                    neighbors = Counter(row[:7]).most_common()

                    key_neighbors = [value[0] for value in neighbors]
                    freq_neighbors = [value[1] for value in neighbors]

                    if k in freq_neighbors:
                        # Get indexes where the K is compatible
                        ind.append(index)

                        # Get the neighbors
                        values = key_neighbors[0]
                        majorities.append([values])

                        # Get minority classes
                        minor = list(key_neighbors[1:])
                        minors.append([minor])

                # Filter the competences and original data
                filter_df = data_classe[data_classe.index.isin(ind)]
                size_filter = len(filter_df)

                if not filter_df.empty:
                    # Adding minority classes in RoC on dataframe
                    filter_df.loc[:,'Minor'] = minors
                    filter_df.loc[:,'Major'] = majorities

                    # Get competences from minority and majority predictions
                    minor_target = filter_df[filter_df.apply(lambda x: x['Target'] in x['Minor'], axis=1)]
                    major_target = filter_df[filter_df['Target'] == filter_df['Major']]
                    outer_target = filter_df.loc[set(filter_df.index) - set(minor_target.index) - set(major_target.index)]

                    means_minor = dataframe_pred(minor_target, minor_target.columns, "Minor").sum(axis=0)/size_filter
                    means_major = dataframe_pred(major_target, major_target.columns, "Major").sum(axis=0)/size_filter
                    means_outer = dataframe_pred(outer_target, outer_target.columns, "Outer").sum(axis=0)/size_filter

                    # Concat hits and errors for each classifier
                    competences_df = pd.concat([means_minor, means_major, means_outer], axis=1)
                    competences_df.columns = ["Minor", "Major", "Outer"]

                    competences_df.replace(np.NaN, 0, inplace=True)

                    competences_df.plot(kind='bar')
                    plt.title("Technique: {} - Noise {} - Classe: {} - Neighbors {}".format(technique, noise, classe, k))
                    plt.ylabel("Mean of Competence (Correctly Predicted)")
                    plt.xlabel("Base Classifiers")
                    # plt.show()
                    save_pdf(plt, output_path + "Competence/"+technique+"/", dataset + "_" + technique + "_" + noise + "_classe_" + str(classe) + "_K_" + str(k))

                else:
                    print("There's no neighbors.")

def target_neighbors(path, dataset, gen, technique, ks, classe):
    folder = input_path + gen + "/" + dataset + "/" + path + "/" +dataset + "_" + technique + "_" + path + "_Noise_"

    for k in ks:
        print("K: ", k)
        noise_df = pd.DataFrame(index=["Minor", "Major", "Outer"], columns=noise_params)

        for noise in noise_params:
            print("Noise: {}".format(noise))
            # Read the data
            data = pd.read_csv(folder+noise+".csv")
            data.drop(data.columns[0], axis=1, inplace=True)
                
            # Filter dataset based on class
            data_classe = data[data['Target'] == classe]
            indexes = data_classe.index
                
            # Get neighbors classes
            neighbors_df = data_classe.filter(like='K', axis=1)

            minors = [] # Relative Minority Class
            majorities = [] # Reltive Majority Class
            ind = [] # Index of nieghbors

            # Get indexes and values for each neighbor
            for index, row in zip(indexes, neighbors_df.values):
                neighbors = Counter(row[:7]).most_common()

                key_neighbors = [value[0] for value in neighbors]
                freq_neighbors = [value[1] for value in neighbors]

                if k in freq_neighbors:
                    # Get indexes where the K is compatible
                    ind.append(index)

                    # Get the neighbors
                    values = key_neighbors[0]
                    majorities.append([values])

                    # Get minority classes
                    minor = list(key_neighbors[1:])
                    minors.append([minor])

            # Filter the competences and original data
            filter_df = data_classe[data_classe.index.isin(ind)]

            if not filter_df.empty:
                # Adding minority classes in RoC on dataframe
                filter_df.loc[:,'Minor'] = minors
                filter_df.loc[:,'Major'] = majorities

                # Get competences from minority and majority predictions
                minor_target = filter_df[filter_df.apply(lambda x: x['Target'] in x['Minor'], axis=1)]
                major_target = filter_df[filter_df['Target'] == filter_df['Major']]
                outer_target = filter_df.loc[set(filter_df.index) - set(minor_target.index) - set(major_target.index)]

                minor_pred = dataframe_pred(minor_target, minor_target.columns, "Minor")
                major_pred = dataframe_pred(major_target, major_target.columns, "Major")
                outer_pred = dataframe_pred(outer_target, outer_target.columns, "Outer")

                noise_df.loc['Minor', noise] = get_perc(minor_pred, minor_target)
                noise_df.loc['Major', noise] = get_perc(major_pred, major_target)
                noise_df.loc['Outer', noise] = get_perc(outer_pred, outer_target)

        noise_df.replace(np.NaN, 0, inplace=True)

        noise_df.T.plot(kind='bar')
        plt.title("Technique: {} - Classe: {} - Neighbors {}".format(technique, classe, k))
        plt.ylabel("Accuracy")
        plt.xlabel("Noise Level")
        # plt.show()
        save_pdf(plt, output_path + "Competence/Accuracy/", dataset + "_" + technique + "_classe_" + str(classe) + "_Neighbors_" + str(k))    


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
    
    data_noises.T.plot(kind='bar')
    plt.xlabel("Noise level")
    plt.ylabel("Accuracy")
    plt.xticks(rotation='horizontal')
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=len(class_match.index))
    # plt.show()
    save_pdf(plt, output_path, dataset + "_" + technique)


def accuracy_neighbor_per_noise(gen, dataset, technique, classe):
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
            
            if len(data_neighbors) > 0:
                accuracy = (len(neighbors_match)/len(data_neighbors))*100
            else:
                accuracy = "No neighbors"
            
            neigh_noise.set_value(str(noise), str(k), accuracy)

    return neigh_noise.T


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

    for gen in gen_methods:
        print("Generation Method: ", gen)
        for dataset in datasets:
            print("Dataset: ", dataset)
            for technique in techniques:
                competence_neighbors("Competence", dataset, gen, technique, range(5,7))
                # class_per_noise_technique(gen, dataset, technique)
                # for noise in noise_params:
                # for classe in range(0,5): #Minority and Majority Class
                #     print("Classe: ", classe)
                #     target_neighbors("Competence", dataset, gen, technique, range(5,7), classe)
                #     neigh = accuracy_neighbor_per_noise(gen, dataset, technique, classe)
                #     print(neigh)
                        
