import os
import numpy as np
import pandas as pd
from multiprocessing import Pool

# Classifier for comparison
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron

# Generation Pool techniques
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from deslib.util.sgh import SGH

# Selection phase (DCS)
from deslib.dcs.ola import OLA
from deslib.dcs.lca import LCA
from deslib.dcs.rank import Rank
from deslib.dcs.mcb import MCB

# Selection phase (DES)
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.des_mi import DESMI
from deslib.des.des_knn import DESKNN
from deslib.des.des_clustering import DESClustering
from deslib.des.des_p import DESP

# Imbalancing Learning
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import RandomUnderSampler

# Baseline Methods
from deslib.static.oracle import Oracle

# Common functions
from utils import gen_ensemble, ds_ensemble, balance_dataset
from utils import load_dataset, build_results_df, save_results_df

# Metrics
from metrics import *

# Plot Confusion Matrix
from graphs_plotter import plot_confusion_matrix

# Number of trees for each fold in ['HH103', 'HH124', 'HH129', 'Kyoto2008','Kyoto2009Spring']
params = [[70, 80, 80, 80, 50, 90], [60, 60, 30, 40, 60, 50], [
    90, 80, 80, 90, 80, 90], [50, 60, 20, 20, 20, 60], [100, 80, 90, 90, 90, 80]]


def process(args):
    X_train = args['X_train']
    y_train = args['y_train']
    X_test = args['X_test']
    y_test = args['y_test']
    fold_name = args['fold_name']
    method = args['ds_method']
    gen_method = args['gen_method']
    imb_method = args['imb_method']
    params_rf = args['params']

    if imb_method:
        # In cases where the imbalanced learning techniques have been used
        X_train, y_train = balance_dataset(X_train, y_train, imb_method)

    # Generation method
    base = Perceptron(max_iter=1, n_jobs=-1)
    n_estimators = 100
    pool_clf = gen_ensemble(X_train, y_train, gen_method, base, n_estimators)

    # Evaluation considering Random Forest
    if method == RandomForestClassifier:
        ensemble = RandomForestClassifier(n_estimators=params_rf)
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)

    # Dynamic Selection techniques
    else:
        ensemble = ds_ensemble(X_train, y_train, pool_clf, method)

        # In prediction phase, two options: Oracle and DS techniques
        if method == Oracle:
            predictions = ensemble.predict(X_test, y_test)
        else:
            predictions = ensemble.predict(X_test)

    # Metrics
    mfm = multi_label_Fmeasure(y_test, predictions)
    gmean = geometric_mean(y_test, predictions, "multiclass")
    conf_matrix = confusion_matrix_score(y_test, predictions)
    acc_by_class = accuracy_by_class(y_test, predictions)
    acc = accuracy(y_test, predictions)
    prec = precision(y_test, predictions, "macro")
    rec = recall(y_test, predictions, "macro")
    fmeasure = fmeasure_score(y_test, predictions, "macro")

    metrics = [mfm, gmean, acc, prec, rec, fmeasure, acc_by_class]

    return [fold_name, conf_matrix, metrics, predictions]


def experiment(folds, iteration, activities_list, labels_dict, dyn_selector, noise, gen_method, dataset, imb_method):
    pool = Pool(5)
    jobs = []
    # Y_Test = []

    # Name
    gen_method_name = str(gen_method).split('.')[-1].split('\'')[0]
    dyn_method_name = str(dyn_selector).split('.')[-1].split('\'')[0]
    imb_method_name = str(imb_method).split('.')[-1].split('\'')[0]

    for f, fold in enumerate(folds):
        args = {'X_train': np.array(fold.xTrain)}

        # Brew requires numeric class labels
        y_train = np.array(fold.yTrains[noise])
        args['y_train'] = np.array([labels_dict.get(x) for x in y_train])
        # X test
        args['X_test'] = np.array(fold.xTest)

        # Brew requires numeric class labels
        y_test = np.array(fold.yTest)
        args['y_test'] = np.array([labels_dict.get(x) for x in y_test])
        # Y_Test.append(args['y_test'])

        args['fold_name'] = 'Fold_' + str(f + 1)
        args['noise'] = noise
        args['ds_method'] = dyn_selector
        args['gen_method'] = gen_method
        args['imb_method'] = imb_method
        args['params'] = params[iteration][f]
        jobs.append(args)

    results = list(map(process, jobs)) # Metrics

    folds_name = [result.pop(0) for result in results]
    # Get predictions
    predictions = np.concatenate([result.pop(-1) for result in results], axis=0)

    # if dyn_selector != RandomForestClassifier:
    #     # Competence level for each activity measured for each classifier
    #     competences = [result.pop(-1) for result in results]

    #     # Due to the different amount of classifiers for fold
    #     dfs = []
    #     for comp in competences:
    #         df = pd.DataFrame(comp, columns=["B"+str(i)
    #                                          for i in range(len(comp[0]))])
    #         dfs.append(df)

    #     # Concat competence folds
    #     comp_df = pd.concat(dfs, axis=0)
    #     comp_df.replace(np.NaN, 0, inplace=True)

    # Get all roc indexes
    # neighbors = np.concatenate([result.pop(-1) for result in results], axis=0)

    #     # Columns for neighbors
    #     cols = ["K"+str(i) for i in range(1, 8)]
    #     neighbors_df = pd.DataFrame(neighbors, columns=cols)

    #     # Get all Targets
    #     Y_Test = np.concatenate(Y_Test, axis=0)

    #     # Concat neighbors and competences dataframes
    #     final_df = neighbors_df.merge(
    #         comp_df, left_index=True, right_index=True, how='inner')
    #     # Adding Predictions and Target
    #     final_df["Predictions"] = predictions
    #     final_df["Target"] = Y_Test
    #     save_results_df(final_df, dataset + "/Competence", str(imb_method).split('.')[-1].split('\'')[0], str(gen_method).split(
    #         '.')[-1].split('\'')[0], noise, dataset + "_" + str(dyn_selector).split('.')[-1].split('\'')[0] + "_Competence")

    # else:
    confusion_mx = [result.pop(0) for result in results]

    # Due to the different amount of classifiers for fold
    dfs = []
    for cm in confusion_mx:
        data = pd.DataFrame(0, index=np.arange(len(activities_list)), columns=activities_list)
        for i, row in enumerate(cm):
            columnList = activities_list[:len(row)]
            data.loc[i, columnList] = row

        dfs.append(data)

    # Concat confusion matrix per folds
    comp_df = pd.concat(dfs, axis=0).groupby(level=0).sum()
    cm = comp_df.values

    plot_confusion_matrix(
        cm=cm,
        classes=labels_dict,
        normalize=True,
        title=dataset + ": " + dyn_method_name + " - " + str(noise) + "0%",
        dataset=dataset,
        gen_method=gen_method_name,
        dynamic_method=dyn_method_name,
        noise_level=str(noise)
    )

    pool.close()

    metrics = ['MultiLabel-Fmeasure', 'Gmean', 'Accuracy', 'Precision', 'Recall', 'F1']

    accuracy_class_df, results_df = build_results_df(metrics, folds_name, results, activities_list)
    save_results_df(results_df, dataset, imb_method_name, gen_method_name, noise, dyn_method_name)
    save_results_df(accuracy_class_df, dataset, imb_method_name, gen_method_name, noise, dyn_method_name + "_by_class")


if __name__ == '__main__':
    root = os.path.dirname(__file__)

    gen_methods = [AdaBoostClassifier, BaggingClassifier, SGH]
    ds_methods_dcs = [OLA, LCA, MCB, Rank, RandomForestClassifier]
    ds_methods_des = [KNORAU, KNORAE, DESKNN, DESP, DESMI, DESClustering]

    ds_methods = ds_methods_dcs ++ ds_methods_des
    imb_methods = [SMOTE, RandomOverSampler, RandomUnderSampler, InstanceHardnessThreshold]

    # datasets = ['HH103', 'HH124', 'HH129', 'Kyoto2008', 'Kyoto2009Spring']
    datasets = ['Kyoto2008']

    for iteration, dataset in enumerate(datasets):
        for ds_method in ds_methods:
            for gen_method in gen_methods:
                # for imb_methd in imb_methods:
                if dataset != '.DS_Store':
                    print('\n\n~~ Database : ' + dataset + ' ~~')
                    print('** Gen Method: %s' % (str(gen_method).split('.')[-1].split('\'')[0]))
                    print('** DS Method: %s' % (str(ds_method).split('.')[-1].split('\'')[0] + ' **\n'))

                    folds_list, activities, examples_by_class = load_dataset(dataset)
                    for noise_level in range(0, 6):
                        print('== Noise Parameter --> ' + str(noise_level) + '0% ==\n')
                        experiment(
                            folds_list,
                            iteration,
                            activities,
                            examples_by_class,
                            ds_method,
                            noise_level,
                            gen_method,
                            dataset,
                            None
                        )
