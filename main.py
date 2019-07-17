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
from deslib.dcs.mla import MLA
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.a_priori import APriori
from deslib.dcs.mcb import MCB

#Baseline Methods
from deslib.static.oracle import Oracle

#Generation Pool techniques
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from deslib.util.sgh import SGH

from utils import load_dataset, build_results_df, save_results_df

#Number of trees for each fold in ['HH103', 'HH124', 'HH129', 'Kyoto2008','Kyoto2009Spring']
# params      = [[70,80,80,80,50,90], [60,60,30,40,60,50], [90,80,80,90,80,90], [50,60,20,20,20,60], [100,80,90,90,90,80]]

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

def balance_dataset(tech, X_train, y_train):
    res_dataset = tech(random_state = 42)
    return res_dataset.fit_resample(X_train, y_train)

def compute_accuracy(y_test, y_pred):
    labels = list(set(y_test))

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
    method      = args['ds_method']
    gen_method  = args['gen_method']

    pool_clf = gen_ensemble(X_train, y_train, gen_method)

    if method == "Random Forest":
        ensemble = RandomForestClassifier(n_estimators=100)
        ensemble.fit(X_train, y_train)
        
    else:
        ensemble, indexes_train = ds_ensemble(X_train, y_train, pool_clf, method)
	
    if method == Oracle:
        predictions, neighbors = ensemble.predict(X_test, y_test)
    else:
        predictions, neighbors = ensemble.predict(X_test)

    #Results
    accuracy_by_class, accuracy, \
    precision_micro, \
    recall_micro, \
    f1_score_micro, = compute_accuracy(y_test, predictions)

    return [fold_name, accuracy, accuracy_by_class, precision_micro, recall_micro, f1_score_micro, predictions, neighbors]

def experiment(folds, activities_list, labels_dict, dyn_selector, noise, gen_method):
    pool = Pool(2)
    jobs = []
    Y_Test = []

    for f, fold in enumerate(folds):
        args = {'X_train': np.array(fold.xTrain)}
        y_train = np.array(fold.yTrains[noise])
        # Brew requires numeric class labels
        args['y_train'] = np.array([labels_dict.get(x) for x in y_train])
        args['X_test'] = np.array(fold.xTest)
        y_test = np.array(fold.yTest)
        # Brew requires numeric class labels
        args['y_test'] = np.array([labels_dict.get(x) for x in y_test])

        # Y_Test used to get the targets
        Y_Test.append(args['y_test'])

        args['fold_name'] = 'Fold ' + str(f + 1)
        args['ds_method'] = dyn_selector
        args['gen_method'] = gen_method
        jobs.append(args)

    results = list(map(process, jobs))
    # Get all roc indexes
    neighbors = np.concatenate([result.pop(-1) for result in results], axis=0 )
    predictions = np.concatenate([result.pop(-1) for result in results], axis=0 )

    Y_Test = np.concatenate(Y_Test, axis=0)

    roc_df = pd.DataFrame(neighbors, columns = ["K"+str(i) for i in range(1,8)])
    roc_df["Predictions"] = predictions
    roc_df["Target"] = Y_Test

    pool.close()

    # metrics     = ['Accuracy', 'Precision', 'Recall', 'F1']

    # # Desconsidering Accuracy by class
    # accuracy_class_df, results_df = build_results_df(metrics, results, activities_list)
    # save_results_df(results_df, dataset, str(gen_method).split('.')[-1].split('\'')[0], noise, str(dyn_selector).split('.')[-1].split('\'')[0])
    # save_results_df(accuracy_class_df, dataset, str(gen_method).split('.')[-1].split('\'')[0], noise, str(dyn_selector).split('.')[-1].split('\'')[0] + "_by_class")
    save_results_df(roc_df, dataset, str(gen_method).split('.')[-1].split('\'')[0], noise, dataset + "_" + str(dyn_selector).split('.')[-1].split('\'')[0] +"_Test")

if __name__ == '__main__':
    # Main
    root = os.path.dirname(__file__)
    ds_methods = [OLA, LCA]
    gen_methods = [SGH]
    # datasets = ['HH103', 'HH124', 'HH129', 'Kyoto2008', 'Kyoto2009Spring']
    datasets = ['Kyoto2008']

    for dataset in datasets:
        for ds_method in ds_methods:
            for gen_method in gen_methods:
                if dataset != '.DS_Store':
                    print('\n\n~~~~~~~~~~~ Database : ' + dataset + ' ~~~~~~~~~~~')
                    print('*********** Method : ' + str(ds_method).split('.')[-1].split('\'')[0] + ' ***********\n')
                    print('*********** Gen Method: %s *************' % (str(gen_method).split('.')[-1].split('\'')[0]))
                    folds_list, activities, examples_by_class = load_dataset(dataset)
                    for noise_level in range(0, 6):
                        print('======== Noise Parameter --> ' + str(noise_level) + '0% ========\n')
                        experiment(folds_list, activities, examples_by_class, ds_method, noise_level, gen_method)
