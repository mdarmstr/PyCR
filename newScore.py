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
    #because the index in excel and the index in matrix are different(the index in matrix do not have class, sample name, and class number column)
    DUMMY_COL_INDEX_DIFF = 2
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

    # get sample matrix from file, column is variable and row is sample
    allSampleList,classList = getValFromFile('data/setClass_file.xlsx')
    allSampleList = np.array(allSampleList)
    half_random_sample,rand_idx_list = selectHalfRandom(allSampleList)
    scaled_half_samples,half_mean,half_svd = scale_half_data(half_random_sample)
    scaled_all_samples = scale_all_data(allSampleList,half_mean,half_svd)
    temp_score = calScore(scaled_half_samples, scaled_all_samples)
    oldScore = gen_clust.RunClust(temp_score,classList,2)
    finalOutPutIdx = []
    copy_all_scaled_samples = copy.deepcopy(scaled_all_samples)
    copy_half_scaled_samples = copy.deepcopy(scaled_half_samples)
    delete_diff = 0

    for idx in startNumList:
        scaled_all_samples = np.delete(scaled_all_samples,[(idx-DUMMY_COL_INDEX_DIFF-delete_diff)],1)
        scaled_half_samples = np.delete(scaled_half_samples,[(idx-DUMMY_COL_INDEX_DIFF-delete_diff)],1)
        temp_score = calScore(scaled_half_samples, scaled_all_samples)
        newScore = gen_clust.RunClust(temp_score, classList, 2)
        if newScore >= oldScore:
            oldScore = newScore
            finalOutPutIdx.append(idx)
            delete_diff = delete_diff +1
        if newScore < oldScore:
            insert_all_value = copy_all_scaled_samples[:,(idx-DUMMY_COL_INDEX_DIFF)]
            insert_half_value = copy_half_scaled_samples[:,(idx-DUMMY_COL_INDEX_DIFF)]
            scaled_all_samples = np.insert(scaled_all_samples,(idx-DUMMY_COL_INDEX_DIFF-delete_diff),insert_all_value,axis=1)
            scaled_half_samples = np.insert(scaled_half_samples,(idx-DUMMY_COL_INDEX_DIFF-delete_diff),insert_half_value,axis=1)
    # set threshold incase we dont have enough variables
    finalOutPutIdx = np.array(finalOutPutIdx)
    selected_all_matrix = copy_all_scaled_samples[:, (finalOutPutIdx.astype(int) - DUMMY_COL_INDEX_DIFF)]
    if selected_all_matrix.shape[1] < 2:
        finalOutPutIdx = startNumList[:10]
    finalOutPutIdx = list(finalOutPutIdx)

    for index in endNumList:
        finalOutPutIdx.append(index)
        finalOutPutIdx = np.array(finalOutPutIdx)
        selected_all_matrix = copy_all_scaled_samples[:, (finalOutPutIdx - DUMMY_COL_INDEX_DIFF)]
        selected_half_matrix = copy_half_scaled_samples[:, (finalOutPutIdx - DUMMY_COL_INDEX_DIFF)]
        finalOutPutIdx = list(finalOutPutIdx)
        temp_score = calScore(selected_half_matrix, selected_all_matrix)
        newScore = gen_clust.RunClust(temp_score, classList, 2)
        if newScore >= oldScore:
            oldScore = newScore
        if newScore < oldScore:
            finalOutPutIdx.remove(index)
    return finalOutPutIdx




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

















