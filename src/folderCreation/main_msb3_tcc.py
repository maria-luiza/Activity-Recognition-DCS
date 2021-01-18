import os
import pickle
import time
import traceback
import random


import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from TCC.fold_msb3_tcc import fold_msb3_tcc
from TCC.result_msb3_tcc import result_msb3_tcc

np.set_printoptions(threshold=np.nan)

SVM = 0
NAIVE_BAYES = 1
RANDOM_FOREST = 2
KNN = 3
DECISION_TREE = 4
MLP = 5

algorithm = SVM


def runSVM(xTrain,yTrain):
    svmMachine = SVC(kernel='rbf')
    svmMachine.fit(xTrain,yTrain)
    return svmMachine


def runNaiveBayes(xTrain,yTrain):
    naiveBayesMachine = GaussianNB()
    naiveBayesMachine.fit(xTrain,yTrain)
    return naiveBayesMachine


def runRandomForest(xTrain,yTrain):
    randomForestMachine = RandomForestClassifier(n_jobs=2)
    randomForestMachine.fit(xTrain,yTrain)
    return randomForestMachine

def runKNN(xTrain,yTrain):
    knnMachine = KNeighborsClassifier(n_neighbors=3)
    knnMachine.fit(xTrain,yTrain)
    return knnMachine

def runDecisionTree(xTrain,yTrain):
    decisionTreeMachine = tree.DecisionTreeClassifier()
    decisionTreeMachine.fit(xTrain,yTrain)
    return decisionTreeMachine

def runMLP(xTrain,yTrain):
    MLPMachine = MLPClassifier()
    MLPMachine.fit(xTrain,yTrain)
    return MLPMachine

# def getActivityCountDict(array):
#     unique, counts = np.unique(array, return_counts=True)
#     return dict(zip(unique, counts))

def calculateGeneralAccuracy(yTest, predictions):
    result = accuracy_score(yTest, predictions)
    return result


def generateRandomLabels(yTrain0, uniqueActivitiesList):
    yTrain = []
    size = len(yTrain0)
    yTrain.append(yTrain0)
    yTrainCopy = yTrain0
    numChanges = int(abs(size / 10))

    for times in range(0,5):
        yTrainCopy = yTrainCopy.copy()
        for change in range(0,numChanges):
            index = random.randint(0, size-1)
            haha = str(random.choice(uniqueActivitiesList))
            yTrainCopy[index] = haha
        yTrain.append(yTrainCopy)
    print('fold created!')
    return yTrain



def createOrGetExistingFolds(file):
    folds = []
    if (os.path.exists("folds/" + file)): #getExisting
        with open('folds/' + file, 'rb') as f:
            folds = pickle.load(f)
            uniqueActivitiesList = pickle.load(f)
            #print(",".join(map(str, uniqueActivitiesList)))
            f.close()
    else: #Create
        with open('inputData/'+file,'rb') as f:
            print(file)
            x = pickle.load(f)
            y = pickle.load(f)
            uniqueActivitiesList = pickle.load(f)
            if (x != [] and y != []):
                rowNum = len(x)
                colNum = len(x[0])
                kFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)

                for foldNumber,(trainIndexes, testIndexes) in enumerate(kFold.split(x,y)):
                    xTrain = []
                    xTest = []
                    yTest = []

                    trainingString = ""
                    testString = ""

                    yTrain0 = []
                    for index in trainIndexes:
                        xTrain.append(x[index])
                        yTrain0.append(y[index])
                        trainingString = trainingString + ",".join(map(str, x[index])) + "," + y[index] + "\n"
                    with open('foldsStr/' + file + "_Fold_" + str(foldNumber) + "_Training.csv", 'w') as ft:
                        ft.write(trainingString)
                        ft.close()

                    for index in testIndexes:
                        xTest.append(x[index])
                        yTest.append(y[index])
                        testString = testString + ",".join(map(str, x[index])) + "," + y[index] + "\n"
                    with open('foldsStr/' + file + "_Fold_" + str(foldNumber) + "_Testing.csv", 'w') as ft2:
                        ft2.write(testString)
                        ft2.close()

                    yTrains = generateRandomLabels(yTrain0, uniqueActivitiesList)

                    fold = fold_msb3_tcc(xTrain, yTrains, xTest, yTest)
                    folds.append(fold)
                with open('folds/' + file, 'wb') as fp:
                    pickle.dump(folds, fp)
                    pickle.dump(uniqueActivitiesList,fp)
                    fp.close()
            f.close()
    return [folds, uniqueActivitiesList]

def calculateRates(yTest, predictions):
    accuracyPerActivity = []
    precisionPerActivity = []
    recallPerActivity = []
    fmeasurePerActivity = []

    activityDictPredictions = np.unique(predictions)
    for activity in activityDictPredictions:
        if (activity != 'Other'):
            tp=0    # Predict C ; Real C
            fp=0    # Predict C ; Real !C
            fn=0    # Predict !C ; Real C
            tn=0    # Predict !C ; Real !C
            accuracy=0
            precision=0
            recall=0
            fmeasure=0
            for i in range(0, len(predictions)):
                if (predictions[i] == activity):
                    if (activity == yTest[i]):
                        tp += 1
                    else:
                        fp += 1
                else:
                    if(activity != yTest[i]):
                        tn += 1
                    else:
                        fn += 1

            if (tp+tn+fn+fp != 0):
                accuracy = (tp+tn)/(tp+tn+fn+fp)
            if (tp+fp != 0):
                precision = tp/(tp+fp)
            if (tp+fn != 0):
                recall = tp/(tp+fn)
            if (precision+recall != 0):
                fmeasure = (2 * precision * recall)/(precision + recall)

            accuracyPerActivity.append(accuracy)
            precisionPerActivity.append(precision)
            recallPerActivity.append(recall)
            fmeasurePerActivity.append(fmeasure)

    foldAccuracy = np.mean(accuracyPerActivity)
    foldPrecision = np.mean(precisionPerActivity)
    foldRecall = np.mean(recallPerActivity)
    foldFmeasure = np.mean(fmeasurePerActivity)


    return foldAccuracy,foldPrecision, foldRecall, foldFmeasure

def evaluateFold(yTest):
    predictions = machine.predict(xTest)
    predictions = np.array(predictions)
    yTest = np.array(yTest)
    foldAccuracy,foldPrecision, foldRecall, foldFmeasure = calculateRates(yTest, predictions)
    foldGeneralAccuracy = calculateGeneralAccuracy(yTest, predictions)
    foldConfusionMatrix = confusion_matrix(yTest, predictions, labels=uniqueActivitiesList)

    generalAccuracyPerFold.append(foldGeneralAccuracy)
    accuracyPerFold.append(foldAccuracy)
    precisionPerFold.append(foldPrecision)
    recallPerFold.append(foldRecall)
    fmeasurePerFold.append(foldFmeasure)
    confusionMatrixPerFold.append(foldConfusionMatrix)
    printAndSaveResults(i,str(foldGeneralAccuracy),str(foldAccuracy),str(foldPrecision),str(foldRecall),str(foldFmeasure), foldConfusionMatrix)


def printAndSaveResults(i, generalAccuracy, accuracy, precision, recall, fmeasure, confusionMatrix):
    if (i != -1):
        print("* Fold : " + str(i))
    else:
        print("@ Result")
        print(uniqueActivitiesList)
    print("General Accuracy : " + generalAccuracy)
    print("Accuracy : " + accuracy)
    print("Precision : " + precision)
    print("Recall : " + recall)
    print("F-Measure : " + fmeasure)
    print("Confusion Matrix")
    #print(uniqueActivitiesList)
    print(confusionMatrix)
    result = result_msb3_tcc(i, generalAccuracy, accuracy, precision, recall, fmeasure, confusionMatrix)
    resultsArray.append(result)
    print()


def getFoldsMean():
    generalAccuracyMean = np.mean(generalAccuracyPerFold)
    accuracyMean = np.mean(accuracyPerFold)
    precisionMean = np.mean(precisionPerFold)
    recallMean = np.mean(recallPerFold)
    fmeasureMean = np.mean(fmeasurePerFold)
    confusionMatrixMean = np.mean(confusionMatrixPerFold,axis=0)
    printAndSaveResults(-1, str(generalAccuracyMean), str(accuracyMean), str(precisionMean), str(recallMean), str(fmeasureMean), confusionMatrixMean)


if __name__ == '__main__':
    inputDataFiles = os.listdir("inputData")
    for file in inputDataFiles:
        if file != ".DS_Store":
            try:
                print("~~~~~~~~~~~ Database : " + file + " ~~~~~~~~~~~\n")
                folds,uniqueActivitiesList = createOrGetExistingFolds(file)
                for noise in range(0,6):
                    generalAccuracyPerFold = []
                    accuracyPerFold = []
                    precisionPerFold = []
                    recallPerFold = []
                    fmeasurePerFold = []
                    confusionMatrixPerFold = []
                    i = 0

                    print("======== Noise Parameter --> " + str(noise) + "0% ========")

                    resultsArray = []
                    for fold in folds:
                        i=i+1
                        xTrain = fold.xTrain
                        yTrain = fold.yTrains[noise]
                        xTest = fold.xTest
                        yTest = fold.yTest

                        if (algorithm == SVM):
                            machine = runSVM(xTrain,yTrain)
                        elif (algorithm == NAIVE_BAYES):
                            machine = runNaiveBayes(xTrain,yTrain)
                        elif (algorithm == RANDOM_FOREST):
                            machine = runRandomForest(xTrain,yTrain)
                        elif (algorithm == KNN):
                            machine = runKNN(xTrain,yTrain)
                        elif (algorithm == DECISION_TREE):
                            machine = runDecisionTree(xTrain,yTrain)
                        elif (algorithm == MLP):
                            machine = runMLP(xTrain,yTrain)

                        evaluateFold(yTest)

                    getFoldsMean()
                    with open('results/'+file + "_" + str(algorithm)+"_Noise_0"+str(noise),'wb') as fp:
                        pickle.dump(resultsArray,fp)
                        pickle.dump(uniqueActivitiesList, fp)
                        fp.close()
            except:
                print('Exception : ' + file)
                traceback.print_exc()
                pass




