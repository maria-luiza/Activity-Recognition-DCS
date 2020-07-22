import os
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from multiprocessing import Pool

# Metrics for evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

# Base Classifier
from sklearn.linear_model import Perceptron

# Classifier for comparison
from sklearn.ensemble import RandomForestClassifier

# Selection phase
# DCS
from deslib.dcs.ola import OLA
from deslib.dcs.lca import LCA
from deslib.dcs.rank import Rank
from deslib.dcs.mcb import MCB

# Baseline Methods
from deslib.static.oracle import Oracle

# Generation Pool techniques
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from deslib.util.sgh import SGH

from utils import load_dataset, build_results_df, save_results_df

# Imbalancing Learning
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import RandomUnderSampler

# Plot Confusion Matrix
from plotter import plot_confusion_matrix

# Number of trees for each fold in ['HH103', 'HH124', 'HH129', 'Kyoto2008','Kyoto2009Spring']
params = [[70, 80, 80, 80, 50, 90], [60, 60, 30, 40, 60, 50], [
    90, 80, 80, 90, 80, 90], [50, 60, 20, 20, 20, 60], [100, 80, 90, 90, 90, 80]]


def gen_ensemble(X_train, y_train, gen_method):
    # Calibrating Perceptrons to estimate probabilities
    base_clf = CalibratedClassifierCV(Perceptron(max_iter=1, n_jobs=-1))
    pool_clf = gen_method(base_clf, n_estimators=100)
    pool_clf.fit(X_train, y_train)
    return pool_clf


def ds_ensemble(X_train, y_train, pool_clf, dyn_sel_method):
    ds_met = dyn_sel_method(pool_clf, selection_method='best')
    indexes = ds_met.fit(X_train, y_train)
    return ds_met, indexes


def balance_dataset(X_train, y_train, balanc):
    res_dataset = balanc()
    return res_dataset.fit_resample(X_train, y_train)


def compute_accuracy(y_test, y_pred):
    labels = list(set(y_test))

    # Evaluate scores - Accuracy, Precision, Recall, and F1 Score
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    accuracy_by_class = {}
    for label, acc in zip(labels, conf_matrix.diagonal() / conf_matrix.sum(axis=1)):
        accuracy_by_class[label] = acc

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy_by_class, accuracy,\
        precision, \
        recall, \
        f1, \
        conf_matrix


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
        # Imbalanced Learning techiques
        X_train, y_train = balance_dataset(X_train, y_train, imb_method)

    # Generation method
    pool_clf = gen_ensemble(X_train, y_train, gen_method)

    # Evaluation considering Random Forest
    if method == RandomForestClassifier:
        neighbors_test, competences = [], []
        ensemble = RandomForestClassifier(n_estimators=params_rf)
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)

    # Dynamic Selection techniques
    else:
        ensemble, _ = ds_ensemble(
            X_train, y_train, pool_clf, method)
        # In prediction phase, two options: Oracle and DS techniques
        if method == Oracle:
            predictions, neighbors_test = ensemble.predict(
                X_test, y_test)
        else:
            predictions, neighbors_test = ensemble.predict(X_test)

    # Results
    accuracy_by_class, accuracy, \
        precision_micro, \
        recall_micro, \
        f1_score_micro, \
        confMatrix = compute_accuracy(y_test, predictions)

    return [fold_name, accuracy, accuracy_by_class, precision_micro, recall_micro, f1_score_micro, confMatrix, neighbors_test, predictions]


def experiment(folds, iteration, activities_list, labels_dict, dyn_selector, noise, gen_method, dataset, imb_method):
    pool = Pool(5)
    jobs, Y_Test = [], []

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
        Y_Test.append(args['y_test'])

        args['fold_name'] = 'Fold_' + str(f + 1)
        args['noise'] = noise
        args['ds_method'] = dyn_selector
        args['gen_method'] = gen_method
        args['imb_method'] = imb_method
        args['params'] = params[iteration][f]
        jobs.append(args)

    results = list(map(process, jobs))

    # Get all predictions
    predictions = np.concatenate([result.pop(-1)
                                  for result in results], axis=0)

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
    results = [result[:-1] for result in results]

    confusion_mx = [result.pop(-1) for result in results]

    # Due to the different amount of classifiers for fold
    dfs = []
    for cm in confusion_mx:
        data = pd.DataFrame(0, index=np.arange(len(activities_list)), columns=activities_list)
        for i, row in enumerate(cm):
            columnList = activities_list[:len(row)]
            data.loc[i,columnList] = row
        
        dfs.append(data)

    # Concat competence folds
    comp_df = pd.concat(dfs, axis=0)

    cm = comp_df.groupby(level=0).sum(axis=1)
    cm = cm.values

    plot_confusion_matrix(
        cm = cm,
        classes = labels_dict,
        normalize=True,
        title= dataset + ": " + str(dyn_selector).split('.')[-1].split('\'')[0] + " - " + str(noise) + "0%",
        dataset = dataset,
        gen_method = str(gen_method).split('.')[-1].split('\'')[0],
        dynamic_method = str(dyn_selector).split('.')[-1].split('\'')[0],
        noise_level = str(noise)
    )

    pool.close()
    # metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

    # accuracy_class_df, results_df = build_results_df(
    #     metrics, results, activities_list)
    # save_results_df(results_df, dataset, str(imb_method).split('.')[-1].split('\'')[0], str(
    #     gen_method).split('.')[-1].split('\'')[0], noise, str(dyn_selector).split('.')[-1].split('\'')[0])
    # save_results_df(accuracy_class_df, dataset, str(imb_method).split('.')[-1].split('\'')[0], str(gen_method).split(
    #     '.')[-1].split('\'')[0], noise, str(dyn_selector).split('.')[-1].split('\'')[0] + "_by_class")


if __name__ == '__main__':
    # Main
    root = os.path.dirname(__file__)
    gen_methods = [SGH]
    # imb_methods = [SMOTE, RandomOverSampler,
    #                RandomUnderSampler, InstanceHardnessThreshold]
    ds_methods = [OLA, LCA, MCB, Rank, RandomForestClassifier]
    # datasets = ['HH103', 'HH124', 'HH129', 'Kyoto2008', 'Kyoto2009Spring']
    datasets = ['HH124']

    for iteration, dataset in enumerate(datasets):
        for ds_method in ds_methods:
            for gen_method in gen_methods:
                # for imb_methd in imb_methods:
                if dataset != '.DS_Store':
                    print('\n\n~~~~~~~~~~~ Database : ' +
                          dataset + ' ~~~~~~~~~~~')
                    print('*********** Method : ' + str(ds_method).split('.')
                          [-1].split('\'')[0] + ' ***********\n')
                    print('*********** Gen Method: %s *************' %
                          (str(gen_method).split('.')[-1].split('\'')[0]))
                    # print('*********** Imbalanced Method: %s *************' %
                    #       (str(imb_methd).split('.')[-1].split('\'')[0]))
                    folds_list, activities, examples_by_class = load_dataset(dataset)
                    for noise_level in range(0, 6):
                        print('======== Noise Parameter --> ' +
                              str(noise_level) + '0% ========\n')
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
