from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score

"""
    Metrics used to measure the classification performed by the techniques.
"""


def accuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred)


def accuracy_by_class(y_test, y_pred):
    labels = list(set(y_test))
    conf = confusion_matrix(y_test, y_pred)

    accuracy_by_class = {}
    for label, acc in zip(labels, conf.diagonal() / conf.sum(axis=1)):
        accuracy_by_class[label] = acc

    return accuracy_by_class


def confusion_matrix_score(y_test, y_pred):
    labels = list(set(y_test))

    return confusion_matrix(y_test, y_pred, labels=labels)


def fmeasure_score(y_test, y_pred, average):
    return f1_score(y_test, y_pred, average=average)


def recall(y_test, y_pred, average):
    return recall_score(y_test, y_pred, average=average)


def precision(y_test, y_pred, average):
    return precision_score(y_test, y_pred, average=average)


def multi_label_Fmeasure(y_test, y_pred):
    """
        MFM is an adaptation of Fmeasure to multiclass problems.
        It corresponds to a sum of Fmeasure for each class divided by the number of classes.
    """
    size_classes = len(list(set(y_test)))  # Calculates the number of classes available on dataset
    fmeasure = fmeasure_score(y_test, y_pred, None)  # The scores for each class are returned

    return sum(fmeasure) / size_classes


def geometric_mean(y_test, y_pred, average):
    """
        Geometric mean (Gmean)
        For binary classification G-mean is the squared root of the product of the sensitivity
        and specificity. For multi-class problems it is a higher root of the product of sensitivity for each class.
    """
    return geometric_mean_score(y_test, y_pred, average=average)
