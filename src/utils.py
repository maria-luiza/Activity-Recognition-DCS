import os
import docx
import shlex
import pandas as pd
import numpy as np
import pickle
from matplotlib.backends.backend_pdf import PdfPages

# Metrics for evaluation
from sklearn.calibration import CalibratedClassifierCV

root = "../"

def load_folds(file_name):
    file_path = root + '/folds/' + file_name
    with open(file_path, 'rb') as file:
        _fold = pickle.load(file)
        unique_activities_list = pickle.load(file)
        file.close()
    return _fold, unique_activities_list


def load_dataset(dataset_name):
    folds, activities_list = load_folds(dataset_name)
    labels_dict = {act: i for i, act in enumerate(activities_list)}
    activities_list = [labels_dict.get(x) for x in activities_list]
    return folds, activities_list, labels_dict


def get_text(filename):
    doc = docx.Document(filename)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return full_text


def string_to_confusion_matrix(conf_matrix):
    conf_matrix = conf_matrix.split('] [')
    for i in range(len(conf_matrix)):
        conf_matrix[i] = conf_matrix[i].replace('[', '')
        conf_matrix[i] = conf_matrix[i].replace(']', '')
        conf_matrix[i] = ','.join(shlex.split(conf_matrix[i]))
        conf_matrix[i] = conf_matrix[i].split(',')
        conf_matrix[i] = [int(n) for n in conf_matrix[i]]
    conf_matrix = np.array(conf_matrix)
    return conf_matrix


def build_results_df(metrics, folds, results, labels):
    results_df = pd.DataFrame(columns=metrics, index=['Fold_' + str(f) for f in range(1, len(folds) + 1)] + ['Mean'])
    acc_class = pd.DataFrame(columns=labels, index=['Fold_' + str(f) for f in range(1, len(folds) + 1)] + ['Mean'])

    for i, functions in enumerate(results):
        functions = functions[0]
        acc_by_class = functions.pop(-1)
        results_df.loc[folds[i]] = functions

        for key in acc_by_class.keys():
            acc_class.loc[folds[i], key] = acc_by_class[key]

    acc_class.loc['Mean'] = acc_class.mean(axis=0)
    acc_class = acc_class.fillna(0)
    results_df.loc['Mean'] = results_df.mean(axis=0)

    return acc_class, results_df.reindex(['Fold_' + str(f) for f in range(1, 6)] + ['Mean'])


def save_results_df(accuracy_df, dataset, imb_method, gen_method, noise, technique):
    if imb_method == "None":
        imb_method = "imbalanced"

    file_path = root + '/Results/' + '/' + imb_method + '/' + gen_method + '/' + dataset + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    noise = "%02d" % int(noise)
    noise = noise[1] + noise[0]
    accuracy_df.to_csv(file_path + technique + '_Noise_' + noise + '.csv', sep=',')


def save_pdf(plot, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    with PdfPages(path + name + '.pdf') as pdf:
        d = pdf.infodict()
        d['Title'] = 'Results'
        d['Author'] = u'Maria Luiza'
        pdf.savefig(bbox_inches="tight", dpi=100)
        plot.close()


def gen_ensemble(X_train, y_train, gen_method, base, n_estimators, cv):
    # Base Classifier - Perceptron, Decision Tree, etc.
    baseClassifier = base
    # Calibrating Perceptrons to estimate probabilities
    base_clf = CalibratedClassifierCV(baseClassifier, cv=cv)
    # Generation technique used to create the pool
    pool_clf = gen_method(base_clf, n_estimators=n_estimators)
    # Train the classifiers in the pool
    pool_clf.fit(X_train, y_train)

    return pool_clf


def ds_ensemble(X_train, y_train, pool_clf, dyn_sel_method):
    # Dynamic Selection method
    ds_met = dyn_sel_method(pool_clf)
    # Train with DSel = Dtrain
    ds_met.fit(X_train, y_train)

    return ds_met


def balance_dataset(X_train, y_train, balanc):
    # Balance method techniques
    res_dataset = balanc()

    return res_dataset.fit_resample(X_train, y_train)
