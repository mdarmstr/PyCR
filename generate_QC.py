import pandas as pd
import random
import numpy as np
import csv

# generate a QC like data
# INPUT: file name
# OUTPUT: QC data set
def generateQC(classFile, sampleFile):
    sampleList = getValFromFileByRows(sampleFile)
    classList = getValFromFileByCols(classFile)[0]
    unique_class = set(classList)
    unique_class = sorted(list(unique_class))
    classNum = len(unique_class)
    class_trans_dict = {}
    for i in range(classNum):
        class_trans_dict[unique_class[i]] = str(i+1)
    for key in class_trans_dict.keys():
        classList = [sub.replace(key, class_trans_dict[key]) for sub in classList]
    classList = [int(x) for x in classList]
    class_idxs = []
    selected_class = classList[0]
    for c in range(len(classList)):
        if classList[c] == selected_class:
            class_idxs.append(c)
    rand_class_idxs = random.sample(class_idxs, 5)
    sampleList = np.array(sampleList)
    rand_samples = sampleList[rand_class_idxs,:]
    rand_samples = list(rand_samples)
    for s in range(len(rand_samples)):
        rand_samples += list(addFraction(rand_samples[s]))
    return rand_samples

# output csv file content as list by column
# INPUT : file name
# OUTPUT : data list
def getValFromFileByCols(fileName):
    df = pd.read_csv(fileName,header=None)
    row_count, column_count = df.shape
    retData = []
    for col in range(column_count):
        tempData = []
        for row in range(row_count):
            tempData.append(str(df.iloc[row][col]))
        retData.append(tempData)
    return retData

# output csv file content as list by row
# INPUT : file name
# OUTPUT : data list
def getValFromFileByRows(fileName):
    df = pd.read_csv(fileName,header=None)
    row_count, column_count = df.shape
    retData = []
    for row in range(row_count):
        tempData = []
        for col in range(column_count):
            tempData.append(float(df.iloc[row][col]))
        retData.append(tempData)
    return retData

def addFraction(sample):
    temp_sample_list = []
    fraction = 0.1
    for i in range(9):
        temp_frac = fraction*i
        temp_sample_list.append(sample*temp_frac)
    return temp_sample_list


