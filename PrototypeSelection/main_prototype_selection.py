import os
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from multiprocessing import Pool

#Metrics for evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

#Base Classifier
from sklearn.linear_model import Perceptron

#Classifier for comparison
from sklearn.ensemble import RandomForestClassifier

#Selection phase
#DCS
from deslib.dcs.ola import OLA
from deslib.dcs.lca import LCA
from deslib.dcs.rank import Rank
from deslib.dcs.mcb import MCB

#Baseline Methods
from deslib.static.oracle import Oracle

#Generation Pool techniques
from deslib.util.sgh import SGH

from utils import load_dataset, build_results_df, save_results_df

# Prototype Selection
from imblearn.under_sampling._prototype_selection import EditedNearestNeighbours
from imblearn.under_sampling._prototype_selection import CondensedNearestNeighbour
from imblearn.under_sampling._prototype_selection import RepeatedEditedNearestNeighbours
from imblearn.under_sampling._prototype_selection import AllKNN
from imblearn.under_sampling._prototype_selection import TomekLinks
from imblearn.under_sampling._prototype_selection import NeighbourhoodCleaningRule

from collections import defaultdict


#Number of trees for each fold in ['HH103', 'HH124', 'HH129', 'Kyoto2008','Kyoto2009Spring']
params = [[70,80,80,80,50,90], [60,60,30,40,60,50], [90,80,80,90,80,90], [50,60,20,20,20,60], [100,80,90,90,90,80]]

def gen_ensemble(X_train, y_train, gen_method):
    # Calibrating Perceptrons to estimate probabilities
    base_clf = CalibratedClassifierCV(Perceptron(max_iter = 1, n_jobs = -1))
    pool_clf = gen_method(base_clf, n_estimators = 100)
    pool_clf.fit(X_train, y_train)
    return pool_clf

def ds_ensemble(X_train, y_train, pool_clf, dyn_sel_method):
    ds_met   = dyn_sel_method(pool_clf, selection_method='best')
    indexes = ds_met.fit(X_train, y_train)
    return ds_met, indexes

def prototype_selection_dataset(X_train, y_train, proto_sel):
    X_, Y_ = proto_sel().fit_resample(X_train, y_train)
    return X_, Y_

def compute_accuracy(y_test, y_pred):
    labels = list(set(y_test))

    # Evaluate scores - Accuracy, Precision, Recall, and F1 Score
    conf_matrix   = confusion_matrix(y_test, y_pred, labels=labels)
    precision     = precision_score(y_test, y_pred, average='micro')
    recall        = recall_score(y_test, y_pred, average='micro')
    f1            = f1_score(y_test, y_pred, average='micro')
    
    accuracy_by_class = {}
    for label, acc in zip(labels, conf_matrix.diagonal() / conf_matrix.sum(axis=1)):
        accuracy_by_class[label] = acc

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy_by_class, accuracy,\
           precision, \
           recall, \
           f1

def process(args):
    X_train     = args['X_train']
    y_train     = args['y_train']
    X_test      = args['X_test']
    y_test      = args['y_test']
    fold_name   = args['fold_name']
    methods     = args['ds_method']
    pool_clf    = args['pool_classifiers']
    params_rf   = args['params']

    results = {}
    for ds_method in methods:
        # Evaluation considering Random Forest
        if ds_method == RandomForestClassifier:
            competences = []
            ensemble = RandomForestClassifier(n_estimators=params_rf)
            ensemble.fit(X_train, y_train)
            predictions = ensemble.predict(X_test)

        # Dynamic Selection techniques
        else:
            ensemble, neighbors_train = ds_ensemble(X_train, y_train, pool_clf, ds_method)
            if ds_method == Oracle:
                predictions, competences = ensemble.predict(X_test, y_test)
            else:
                predictions, competences = ensemble.predict(X_test)
    
        # Accuracy, Precision, Recall and F1-Score
        accuracy_by_class, accuracy, \
        precision_micro, \
        recall_micro, \
        f1_score_micro, = compute_accuracy(y_test, predictions)

        results[str(ds_method).split('.')[-1].split('\'')[0]] = [
            fold_name,
            accuracy,
            accuracy_by_class,
            precision_micro,
            recall_micro,
            f1_score_micro,
            competences,
            predictions
        ]
    
    return results

def experiment(folds, iteration, activities_list, labels_dict, dyn_selectors, noise, gen_method, proto_selection):
    pool = Pool(5)
    jobs, Y_Test = [], []

    for f, fold in enumerate(folds):
        # X Train
        X_train = np.array(fold.xTrain)
        # Y Train
        y_train = np.array(fold.yTrains[noise])
        y_train = np.array([labels_dict.get(x) for x in y_train])

        # Prototype Selection Technique
        proto_X_train, proto_y_train = prototype_selection_dataset(X_train, y_train, proto_selection)
        # Define the dictionary
        args = {'X_train': proto_X_train}        
        # Brew requires numeric class labels
        args['y_train'] =  proto_y_train

        # Define Ensemble of Classifiers
        pool_clf = gen_ensemble(args['X_train'], args['y_train'], gen_method)

        # X test
        args['X_test'] = np.array(fold.xTest)
        # Brew requires numeric class labels
        y_test = np.array(fold.yTest)
        args['y_test'] = np.array([labels_dict.get(x) for x in y_test])
        Y_Test.append(args['y_test'])

        args['fold_name'] = 'Fold_' + str(f + 1)
        args['noise'] = noise
        # Dynamic Selection techniques that will be evaluated
        args['ds_method'] = dyn_selectors
        # Pool of classifiers for each fold
        args['pool_classifiers'] = pool_clf
        args['params'] = params[iteration][f]
    
        jobs.append(args)

    methods_results = list(map(process, jobs))

    pool.close()

    # Dynamic Selection Names
    ds_methods = [str(dyn_selector).split('.')[-1].split('\'')[0] for dyn_selector in dyn_selectors]

    # Prototype Selection Method
    proto_sel = str(proto_selection).split('.')[-1].split('\'')[0]

    # Generation Method
    gen_met = str(gen_method).split('.')[-1].split('\'')[0]

    # Get all Targets
    Y_Test = np.concatenate(Y_Test, axis=0)

    methods = defaultdict(list)
    for fold in methods_results:
        for ds_method in ds_methods:
            methods[ds_method].append(fold[ds_method])

    for dyn_selector in ds_methods:
        results = methods[dyn_selector]
        # Get all predictions
        predictions = np.concatenate([result.pop(-1) for result in results], axis=0 )
        if dyn_selector != "RandomForestClassifier":
            # Competence level for each activity measured for each classifier
            competences = [result.pop(-1) for result in results]
            # Due to the different amount of classifiers for fold
            dfs = []
            for comp in competences:
                df = pd.DataFrame(comp, columns = ["B"+str(i) for i in range(len(comp[0]))])
                dfs.append(df)
            
            # Concat competence folds
            comp_df = pd.concat(dfs, axis=0)
            comp_df.replace(np.NaN, 0, inplace=True)
            
            # # Get all roc indexes
            # neighbors = np.concatenate([result.pop(-1) for result in results], axis=0 )

            # # Columns for neighbors
            # cols = ["K"+str(i) for i in range(1,8)]
            # neighbors_df = pd.DataFrame(neighbors, columns = cols)

            # # Concat neighbors and competences dataframes
            # final_df = neighbors_df.merge(comp_df, left_index=True, right_index=True, how='inner')
            # Adding Predictions and Target
            comp_df["Predictions"] = predictions
            comp_df["Target"] = Y_Test
            save_results_df(comp_df, dataset + "/Competence", proto_sel, gen_met, noise, dataset + "_" + dyn_selector +"_Competence")
        else:
            results = [result[:-1] for result in results]

        metrics     = ['Accuracy', 'Precision', 'Recall', 'F1']
        accuracy_class_df, results_df = build_results_df(metrics, results, activities_list)
        save_results_df(results_df, dataset, proto_sel, gen_met, noise, dyn_selector)
        save_results_df(accuracy_class_df, dataset, proto_sel, gen_met, noise, dyn_selector + "_by_class")

if __name__ == '__main__':
    # Main
    root = os.path.dirname(__file__)
    gen_methods = [SGH]
    ds_methods = [OLA, LCA, MCB, Rank, RandomForestClassifier]
    proto_selections = [EditedNearestNeighbours, CondensedNearestNeighbour, RepeatedEditedNearestNeighbours, AllKNN, TomekLinks, NeighbourhoodCleaningRule]
    # datasets = ['HH103', 'HH124', 'HH129', 'Kyoto2008', 'Kyoto2009Spring']
    datasets = ['Kyoto2008']

    for iteration, dataset in enumerate(datasets):
        for gen_method in gen_methods:
            for proto_sel in proto_selections:
                print('\n\n~~~~~~~~~~~ Database : ' + dataset + ' ~~~~~~~~~~~')
                print('*********** Gen Method: %s *************' % (str(gen_method).split('.')[-1].split('\'')[0]))
                print('*********** Proto Selection Method: %s *************' % (str(proto_sel).split('.')[-1].split('\'')[0]))
                folds_list, activities, examples_by_class = load_dataset(dataset)
                for noise_level in range(0, 6):
                    print('======== Noise Parameter --> ' + str(noise_level) + '0% ========\n')
                    experiment(
                        folds_list,
                        iteration,
                        activities,
                        examples_by_class,
                        ds_methods,
                        noise_level,
                        gen_method,
                        proto_sel
                    )
                    print("\n========== Done ==========\n")
