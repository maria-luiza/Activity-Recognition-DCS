import os
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from multiprocessing import Pool
from collections import Counter

#Metrics for evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

#Base Classifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
#Classifier for comparison
from sklearn.ensemble import RandomForestClassifier

#Balance dataset
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN

#Selection phase
#DCS
from deslib.dcs.ola import OLA
from deslib.dcs.lca import LCA
from deslib.dcs.rank import Rank
from deslib.dcs.mla import MLA

#DES
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.des_knn import DESKNN
from deslib.des.knop import KNOP

#Baseline Methods
from deslib.static.oracle import Oracle

#Generation Pool techniques
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from deslib.util.sgh import SGH

from utils import load_dataset, build_results_df, save_results_df
from plotter import plot_confusion_matrix

#Number of trees for each fold in ['HH103', 'HH124', 'HH129', 'Kyoto2008','Kyoto2009Spring']
params      = [[70,80,80,80,50,90], [60,60,30,40,60,50], [90,80,80,90,80,90], [50,60,20,20,20,60], [100,80,90,90,90,80]]

def gen_ensemble(X_train, y_train, gen_method):
    # Calibrating Perceptrons to estimate probabilities
    base_clf = CalibratedClassifierCV(Perceptron(max_iter = 1, n_jobs = -1))
    pool_clf = gen_method(base_clf, n_estimators = 100)
    pool_clf.fit(X_train, y_train)
    return pool_clf

def ds_ensemble(X_train, y_train, pool_clf, dyn_sel_method):
    ds_met   = dyn_sel_method(pool_clf)
    ds_met.fit(X_train, y_train)
    return ds_met

def balance_dataset(tech, X_train, y_train):
    res_dataset = tech(random_state = 42)
    return res_dataset.fit_resample(X_train, y_train)

def compute_accuracy(y_test, y_pred):
    labels = list(set(y_test))

    conf_matrix   = confusion_matrix(y_test, y_pred, labels=labels)
    precision     = precision_score(y_test, y_pred, average = 'micro')
    recall        = recall_score(y_test, y_pred, average = 'micro')
    f1            = f1_score(y_test, y_pred, average = 'micro')
    
    accuracy_by_class = {}
    for label, acc in zip(labels, conf_matrix.diagonal() / conf_matrix.sum(axis=1)):
        accuracy_by_class[label] = acc

    accuracy = accuracy_score(y_test, y_pred)

    return conf_matrix, accuracy_by_class, accuracy,\
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
    geneation   = args['gen_method']

    le          = preprocessing.LabelEncoder()
    le.fit(y_train)

    pool_clf = gen_ensemble(X_train, y_train, gen_method)

    print("Nº Estimators: ", pool_clf.n_estimators)

    if method == "Random Forest":
        ensemble = RandomForestClassifier(n_estimators = 100)
        ensemble.fit(X_train, y_train)
        
    else:
        # if ((method == METADES)):
        #     y_train = le.transform(y_train)
        #     y_test  = le.transform(y_test)
    	
        ensemble 	= ds_ensemble(X_train, y_train, pool_clf, method)
	
    if method == Oracle:
        predictions = ensemble.predict(X_test, y_test)
    else:
        predictions = ensemble.predict(X_test)
    
    #Results
    conf_matrix, accuracy_by_class, accuracy, \
    precision_micro, \
    recall_micro, \
    f1_score_micro, = compute_accuracy(y_test, predictions)

    return [fold_name, accuracy, accuracy_by_class, precision_micro, recall_micro, f1_score_micro]

def experiment(folds, activities_list, labels_dict, dyn_selector, noise, gen_method, balance):
    pool = Pool(2)
    jobs = []
    for f, fold in enumerate(folds):
        args = {'X_train': np.array(fold.xTrain)}
        y_train = np.array(fold.yTrains[noise])
        # Brew requires numeric class labels
        args['y_train'] = np.array([labels_dict.get(x) for x in y_train])
        args['X_test'] = np.array(fold.xTest)
        y_test = np.array(fold.yTest)
        # Brew requires numeric class labels
        args['y_test'] = np.array([labels_dict.get(x) for x in y_test])
        args['fold_name'] = 'Fold ' + str(f + 1)
        args['ds_method'] = dyn_selector
        args['gen_method'] = gen_method

        # args['X_train'], args['y_train'] = balance_dataset(balance, args['X_train'], args['y_train'])
        jobs.append(args)

    results = list(map(process, jobs))
    pool.close()

    # print(results)

    # conf_matrix = sum([x.pop(1) for x in results])
    # print(conf_matrix)
    # plot_confusion_matrix(conf_matrix, target_names = activities_list, \
    #                         title = dataset + "__" + str(dyn_selector).split('.')[-1].split('\'')[0] + "__" + str(noise) + "0")

    metrics     = ['Accuracy', 'Precision', 'Recall', 'F1']

    #Desconsidering Accuracy by class
    accuracy_class_df, results_df = build_results_df(metrics, results, activities_list)
    save_results_df(results_df, dataset, str(gen_method).split('.')[-1].split('\'')[0], noise, str(dyn_selector).split('.')[-1].split('\'')[0])
    save_results_df(accuracy_class_df, dataset, str(gen_method).split('.')[-1].split('\'')[0], noise, str(dyn_selector).split('.')[-1].split('\'')[0] + "_by_class")

if __name__ == '__main__':
    # Main
    root = os.path.dirname(__file__)
    # ds_methods  = [Oracle, OLA, LCA, MLA, Rank, KNORAE, KNORAU, DESKNN]
    ds_methods = ['Random Forest']
    gen_methods = [BaggingClassifier, AdaBoostClassifier, SGH]
    # gen_methods = [BaggingClassifier, AdaBoostClassifier]

    datasets    = ['HH103', 'HH124', 'HH129', 'Kyoto2008','Kyoto2009Spring']
    # datasets    = ['HH124', 'HH129', 'Kyoto2008','Kyoto2009Spring']

    for dataset in datasets:
        for ds_method in ds_methods:
            for gen_method in gen_methods:
                if dataset != '.DS_Store':
                    print('\n\n~~~~~~~~~~~ Database : ' + dataset + ' ~~~~~~~~~~~')
                    print('*********** Method : ' + str(ds_method).split('.')[-1].split('\'')[0] + ' ***********\n')
                    print('*********** Gen Method: %s *************' % (str(gen_method).split('.')[-1].split('\'')[0]))
                    folds_list, activities, examples_by_class = load_dataset(dataset)
                    for noise_level in range(0, 6):
                        mean_estimatiors = 0
                        print('======== Noise Parameter --> ' + str(noise_level) + '0% ========\n')
                        experiment(folds_list, activities, examples_by_class, ds_method, noise_level, gen_method, None)