import operator
import pandas as pd
import xlrd
import openpyxl
import copy
from numpy import inf
import sys
from scipy.sparse.linalg import svds
import gen_clust
import numpy as np
from scipy.sparse import csc_matrix
import random
#set the start number and end number
def setNumber(fisherProb,classNum):
    START_NUM = 0.9
    END_NUM = 0.5
    sorted_fisherProb = sorted(fisherProb.items(), key=operator.itemgetter(1), reverse=True)
    # print(sorted_fisherProb)
    startNumList = []
    endNumList = []
    allSampleList_idx = []
    for i in sorted_fisherProb:
        if i[1] > 0.9:
            startNumList.append(i[0])
        if i[1]< 0.9 and i[1]>0.5:
            endNumList.append(i[0])

    for j in sorted_fisherProb:
        allSampleList_idx.append(j[0])

    # get the start compare score
    genfile(allSampleList_idx,"data/setClass_file.xlsx")
    allSampleList,classList = getValFromFile('data/clust.xlsx')
    half_random_sample,rand_idx_list = selectHalfRandom(allSampleList)
    scaled_half_samples,half_mean,half_svd = scale_half_data(half_random_sample)
    scaled_all_samples = scale_all_data(allSampleList,half_mean,half_svd)
    genAllScaleFile(scaled_all_samples.tolist())
    temp_score = calScore(scaled_half_samples, scaled_all_samples)
    oldScore = gen_clust.RunClust(temp_score,classList,2)

def convertToFile(sampleLists):
    wb = xlrd.open_workbook("data/setClass_file.xlsx")
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)
    sample_col = sheet.col_values(0)
    df = pd.DataFrame(sample_col[1:], columns=[sample_col[0]])

    sampleLists = np.transpose(sampleLists)
    loc_counter = 1
    for i in sampleLists:
        tem_col = sheet.col_values(loc_counter+1)
        df.insert(loc_counter, tem_col[0], i, True)
        loc_counter = loc_counter + 1
    df.to_excel("data/scaled_samples.xlsx", index=False)

def genfile(indexList,fileName):
    wb = xlrd.open_workbook(fileName)
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)
    sample_col = sheet.col_values(0)
    df = pd.DataFrame(sample_col[1:], columns=[sample_col[0]])
    class_col =sheet.col_values(2)
    class_name=sheet.col_values(1)
    df.insert(1, class_name[0], class_name[1:], True)
    df.insert(2,class_col[0],class_col[1:],True)
    loc_counter = 3
    for i in indexList:
        tem_col = sheet.col_values(i)
        df.insert(loc_counter,tem_col[0],tem_col[1:],True)
        loc_counter = loc_counter + 1
    df.to_excel("data/clust.xlsx", index=False)

def getScaledValueFromFile(fileName,idxList):
    wb = xlrd.open_workbook(fileName)
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)
    samples = []
    # add all the variables in clust into the variables list
    for i in range(0, sheet.nrows):
        temp_col1 = []
        for z in idxList:
            temp_col1.append(float(sheet.cell_value(i, z)))
        samples.append(temp_col1)
    return np.array(samples)


def genAllScaleFile(indexList):
    wb = openpyxl.Workbook()
    worksheet = wb.active
    for i in range(len(indexList)):
        worksheet.append(indexList[i])
    wb.save('data/scaledclust.xlsx')

def getValFromFile(fileName):
    wb = xlrd.open_workbook(fileName)
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)
    sample_col = sheet.col_values(0)
    df = pd.DataFrame(sample_col[1:], columns=[sample_col[0]])
    sample_class = sheet.col_values(2)
    sample_class = sample_class[1:]
    samples = []
    # add all the variables in clust into the variables list
    for i in range(1, sheet.nrows):
        temp_col1 = []
        for z in range(3, sheet.ncols):
            temp_col1.append(float(sheet.cell_value(i, z)))
        samples.append(temp_col1)
    return samples,sample_class

def scale_all_data(samples,mean,std):
    functionTop = np.subtract(samples, mean)
    scaled_samples = np.divide(functionTop, std)
    for list in scaled_samples:
        list[list==inf] = 10**-12
    scaled_samples = np.nan_to_num(scaled_samples, nan=(10**-12))
    return scaled_samples

def scale_half_data(samples):
    # after get all the selected variables we make them a metrix and calculate the mean
    samples = np.array(samples)
    samples_mean = samples.mean(axis=0)
    samples_std = np.std(samples, axis=0)
    np.set_printoptions(threshold=sys.maxsize)
    functionTop = np.subtract(samples,samples_mean)
    scaled_samples = np.divide(functionTop, samples_std)
    scaled_samples = np.nan_to_num(scaled_samples,nan=(10**-12))
    for list in scaled_samples:
        list[list==inf] = 10**-12

    return scaled_samples, samples_mean, samples_std

# randomly select half variables from the selected_scaled_variables_list
def selectHalfRandom(sample_list):
    idx_list = []
    rand_sample_list = []
    for i in range(len(sample_list)):
        idx_list.append(i)

    total_num = len(sample_list)
    half_num = total_num//2

    rand_idx_list = random.sample(list(idx_list),half_num)
    for idx in rand_idx_list:
        rand_sample_list.append(sample_list[idx])

    return rand_sample_list,rand_idx_list

def calScore(rand_variable_list,all_variable_list):
    # rand_variable_list = csc_matrix(rand_variable_list,dtype=float)
    dummyU,dummyS,V = svds(rand_variable_list, k=2)
    V = np.transpose(V)
    score = np.dot(all_variable_list, V)
    return score

