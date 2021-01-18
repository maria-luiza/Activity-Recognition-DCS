import traceback
from datetime import datetime
import pickle
import os
import numpy as np
global lastActivity



def openFile(dir):
    file = open(dir,"r")
    fileRows = []
    for line in file:
        terms = line.split()
        fileRow = []
        for term in terms:
            fileRow.append(term)
        fileRows.append(fileRow)
    return fileRows


def getMillisecondsFromDate(date, time):
    if('.' not in time):
        time = time + ".00"
    # strptime parses a string representing a time according to a format
    dateAndTime = datetime.strptime(date+" "+time, '%Y-%m-%d %H:%M:%S.%f')
    # dateAndTime.timestamp() Return POSIX timestamp corresponding to the datetime instance
    return dateAndTime.timestamp() * 1000


def getActivityLabel(row, label, wasActivityEnd, activitiesNotFinished):
    if(wasActivityEnd): #was ending an activity previously... it's not anymore. (That class scope is over)
        label = ""
        wasActivityEnd = False
    if (len(row) > 4): # it has an activity
        label = row[4]
        if ('"begin"' in label):
            index = label.find("=")
            label = label[:index]
            activitiesNotFinished.append(label)

        elif('"end"' in label):
            index = label.find("=")
            label = label[:index]
            # Remove last occurrence of the activity at activitiesNotFinished
            activitiesNotFinished.reverse()
            wasActivityEnd = True
            try:
                activitiesNotFinished.remove(label)
            except:
                pass
            activitiesNotFinished.reverse()

        else:
            if (len(row) == 6):
                if (row[5] == "begin"):
                    activitiesNotFinished.append(label)
                elif (row[5] == "end"):
                    # Remove last occurrence of the activity at activitiesNotFinished
                    activitiesNotFinished.reverse()
                    wasActivityEnd = True
                    try:
                        activitiesNotFinished.remove(label)
                    except:
                        pass
                    activitiesNotFinished.reverse()
            else:
                wasActivityEnd = True
    return label, wasActivityEnd



def generateFeatureVector(windowRows, activitiesNotFinished):
    label = ""
    wasActivityEnd = False
    firstRow = windowRows[0]
    #row[0] = yyyy-mm-dd
    #row[1] = hh:mm:ss.ms
    firstSensorTime = getMillisecondsFromDate(firstRow[0], firstRow[1])
    lastRow = windowRows[len(windowRows)-1]
    lastSensorTime = getMillisecondsFromDate(lastRow[0], lastRow[1])
    windowTemporalSpan = (lastSensorTime - firstSensorTime)
    bagOfSensors = getEmptyBagOfSensors().copy()
    for row in windowRows:
        sensor = row[2]
        bagOfSensors[sensor] = bagOfSensors[sensor] + 1
        label,wasActivityEnd = getActivityLabel(row, label, wasActivityEnd, activitiesNotFinished)

    # What was the label computed for the last activity in the window?
    if(label == ""):
        activitiesNotFinishedLength = len(activitiesNotFinished)
        # If there is an incomplete activity or an activity just finished (Gets its class), otherwise lable it 'Other' class
        if (activitiesNotFinishedLength != 0 or wasActivityEnd):
            label = activitiesNotFinished[activitiesNotFinishedLength-1]
        else:
            label = "Other"
    bagOfSensors = [value for(key,value) in sorted(bagOfSensors.items())]
    return [[firstSensorTime, lastSensorTime, windowTemporalSpan, *bagOfSensors], label]


def preprocessData():
    featureVectorArray = []
    labelArray = []
    numberOfWindows = len(fileRows)-windowSize+1
    # Represents activities that began but didn't end
    activitiesNotFinished = []
    for i in range(0,numberOfWindows):
        windowRows = fileRows[i:windowSize+i]
        featureVector = generateFeatureVector(windowRows,activitiesNotFinished)
        featureVectorArray.append(featureVector[0])
        labelArray.append(featureVector[1])
        print(i)
    featureVectorArray = normalizeData(featureVectorArray)
    uniqueActivitiesList = np.unique(np.array(labelArray))
    return [featureVectorArray,labelArray, uniqueActivitiesList]


def getEmptyBagOfSensors():
    sensors = {}
    for row in fileRows:
        sensor = row[2]
        if(sensor not in sensors):
            sensors.update({sensor:0})
    return sensors


def normalizeData(featureVectorArray):
    maxPerColumn = []
    minPerColumn = []
    rangePerColumn = []
    for column in range(0, len(featureVectorArray[0])):
        maxValue = max(row[column] for row in featureVectorArray)
        minValue = min(row[column] for row in featureVectorArray)
        maxPerColumn.append(maxValue)
        minPerColumn.append(minValue)
        rangePerColumn.append(maxValue-minValue)

    for r in range(0,len(featureVectorArray)):
        for c in range(0,len(featureVectorArray[0])):
            featureVectorArray[r][c] = (featureVectorArray[r][c] - minPerColumn[c])/rangePerColumn[c]

    return featureVectorArray


if __name__ == '__main__':
    dataFiles = os.listdir("datasets")
    for file in dataFiles:
        if file != ".DS_Store":
            try:
                fileRows = openFile("datasets/"+file)
                windowSize = 30
                data = preprocessData()
                inputData = data[0]
                labelData = data[1]
                uniqueActivitiesList = data[2]
                print(inputData)
                print(labelData)
                print(uniqueActivitiesList)
                with open('inputData/'+file,'wb') as fp:
                    pickle.dump(inputData,fp)
                    pickle.dump(labelData,fp)
                    pickle.dump(uniqueActivitiesList,fp)
                    fp.close()
            except:
                print('Exception : ' + file)
                traceback.print_exc()
                pass

