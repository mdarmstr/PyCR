import operator
import copy
from numpy import inf
import sys
from scipy.sparse.linalg import svds
import fisherRatio_in
import gen_clust
import numpy as np
import random

def setNumber(classNum, classList, allSampleList, startNum, endNum):
    # get sample matrix from file, column is variable and row is sample
    allSampleList = np.array(allSampleList)
    #get the half randomly selected sample and calculate the fisher ration
    half_random_sample, rand_idx_list = selectHalfRandom(allSampleList)
    half_random_class_list = []
    for i in rand_idx_list:
        half_random_class_list.append(classList[i])
    fisherRatio = fisherRatio_in.cal_ratio(half_random_sample,rand_idx_list, half_random_class_list, classNum)
    sorted_fisherRatio = sorted(fisherRatio.items(), key=operator.itemgetter(1), reverse=True)

    # get the start variable list and end variable list by startNum and end Num
    allSampleList_idx = []
    startNumList = []
    endNumList = []
    for i in sorted_fisherRatio:
        if i[1] > startNum:
            startNumList.append(i[0])
        if i[1]< startNum and i[1]>endNum:
            endNumList.append(i[0])
    for j in sorted_fisherRatio:
        allSampleList_idx.append(j[0])
    #calculate the old score with all the variables inside
    scaled_half_samples,half_mean,half_svd = scale_half_data(half_random_sample)
    scaled_all_samples = scale_all_data(allSampleList,half_mean,half_svd)
    temp_score = calScore(scaled_half_samples, scaled_all_samples)
    oldScore = gen_clust.RunClust(temp_score,classList,2)
    finalOutPutIdx = []
    copy_all_scaled_samples = copy.deepcopy(scaled_all_samples)
    copy_half_scaled_samples = copy.deepcopy(scaled_half_samples)
    delete_diff = 0

    # get rid of the variable form teh start variable list and calculate the score again
    # compare with the the old score
    # if the new score is lower than the old score, we need save the variable in selected variable list
    # if the new score is higher than the old score, we need put the variable back
    for idx in startNumList:
        scaled_all_samples = np.delete(scaled_all_samples, [(idx-delete_diff)], 1)
        scaled_half_samples = np.delete(scaled_half_samples, [(idx-delete_diff)], 1)
        temp_score = calScore(scaled_half_samples, scaled_all_samples)
        newScore = gen_clust.RunClust(temp_score, classList, 2)
        if newScore >= oldScore:
            insert_all_value = copy_all_scaled_samples[:, (idx)]
            insert_half_value = copy_half_scaled_samples[:, (idx)]
            scaled_all_samples = np.insert(scaled_all_samples, (idx - delete_diff),
                                           insert_all_value, axis=1)
            scaled_half_samples = np.insert(scaled_half_samples, (idx - delete_diff),
                                            insert_half_value, axis=1)
        if newScore < oldScore:
            oldScore = newScore
            finalOutPutIdx.append(idx)
            delete_diff = delete_diff + 1

    # set threshold incase we dont have enough variables
    finalOutPutIdx = np.array(finalOutPutIdx)
    selected_all_matrix = copy_all_scaled_samples[:, (finalOutPutIdx.astype(int))]
    if selected_all_matrix.shape[1] < 3:
        finalOutPutIdx = startNumList[:10]

    # calculate the old score with all pre-selected variables
    finalOutPutIdx = np.array(finalOutPutIdx)
    selected_all_matrix = copy_all_scaled_samples[:, (finalOutPutIdx.astype(int))]
    selected_half_matrix = copy_half_scaled_samples[:, (finalOutPutIdx.astype(int))]
    temp_score = calScore(selected_half_matrix, selected_all_matrix)
    oldScore = gen_clust.RunClust(temp_score, classList, 2)
    finalOutPutIdx = list(finalOutPutIdx)

    # add the variable into the selected variable list
    # compare with the the old score
    # if the new score is lower than the old score, we need put the variable back
    # if the new score is higher than the old score, we need save the variable in selected variable list
    for index in endNumList:
        finalOutPutIdx.append(index)
        finalOutPutIdx = np.array(finalOutPutIdx)
        selected_all_matrix = copy_all_scaled_samples[:, (finalOutPutIdx)]
        selected_half_matrix = copy_half_scaled_samples[:, (finalOutPutIdx)]
        finalOutPutIdx = list(finalOutPutIdx)
        temp_score = calScore(selected_half_matrix, selected_all_matrix)
        newScore = gen_clust.RunClust(temp_score, classList, 2)
        if newScore >= oldScore:
            oldScore = newScore
        if newScore < oldScore:
            finalOutPutIdx.remove(index)
    return finalOutPutIdx

# scale all data with provide mean and std
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

















