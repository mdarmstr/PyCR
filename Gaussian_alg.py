import xlrd
import pandas as pd
import numpy as np
import random
import copy
import column_class
import statistics as stat
import scipy.stats
import matplotlib.pyplot as pyplot
from scipy.stats import norm
def gaussian_algorithm(classNum):
    sample_matrix, class_list = getValFromFile("data/setClass_file.xlsx")
    k = 0
    true_means = []
    null_means = []
    while k < 30:
        half_rand_matrix, half_rand_class_list = selectHalfRandom(sample_matrix,class_list)
        classNum_list = []
        for i in range(classNum):
            classNum_list.append(i+1)
        true_dist_classNum = copy.deepcopy(half_rand_class_list)
        null_dist_classNum = []
        for i in range(len(half_rand_class_list)):
            null_dist_classNum.append(random.choice(class_list))
        true_fisherRatio = cal_fish_ratio(half_rand_matrix,true_dist_classNum,classNum)
        null_fisherRatio = cal_fish_ratio(half_rand_matrix,null_dist_classNum,classNum)
        true_mean_fisher_ratio = np.mean(true_fisherRatio)
        null_mean_fisher_ratio = np.mean(null_fisherRatio)
        true_means.append(true_mean_fisher_ratio)
        null_means.append(null_mean_fisher_ratio)
        k = k+1

    m, mMinusH, mPlusH = mean_confidence_interval(true_means, 0.95)
    print(mPlusH)
    print(true_means)

    # print(true_means)
    # true_fish_mean = np.mean(true_means)
    # true_fish_std = np.std(true_means)
    # null_fish_mean = np.mean(null_means)
    # null_fish_std = np.std(null_means)
    #
    # x = np.linspace(0, 2, num=100)
    # pyplot.plot(norm.pdf(x, true_fish_mean, true_fish_std))
    # pyplot.plot(norm.pdf(x, null_fish_mean, null_fish_std))
    # pyplot.show()

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
    samples = np.array(samples)
    return samples, sample_class

def selectHalfRandom(sample_list,class_list):
    sample_idx_list = []
    rand_sample_list = []
    rand_class_list = []
    for i in range(len(sample_list)):
        sample_idx_list.append(i)

    total_sample_num = len(sample_list)
    half_sample_num = total_sample_num//2

    variable_idx_list = []
    rand_variable_list = []

    for i in range(len(sample_list[0])):
        variable_idx_list.append(i)

    total_variable_num = len(sample_list[0])
    half_variable_num = total_variable_num//2


    rand_idx_list = random.sample(list(sample_idx_list), half_sample_num)
    rand_variable_list = random.sample(list(variable_idx_list),half_variable_num)
    for idx in rand_idx_list:
        rand_sample_list.append(sample_list[idx])
        rand_class_list.append(class_list[idx])
    random_sample_matrix = np.array(rand_sample_list)
    final_rand_matrix = random_sample_matrix[:,rand_variable_list]

    return final_rand_matrix,rand_class_list

def cal_mean(sample_list):
    sample_mean_list = []
    for i in sample_list:
        sum_sample = sum(i)
        sample_mean =sum_sample/len(i)
        sample_mean_list.append(sample_mean)
    return sample_mean_list

def cal_fish_ratio(sample_list,class_list,classNum):

    # define a fisher ratio list for all columns with default value 0

    fish_ratio = []
    # for each column sample type we calculate one fisher ratio for one column
    for i in range(len(sample_list[0])):
        #define a data list for all class
        # define a data list contain different class data list
        class_data = []
        for k in range(classNum + 1):
            class_data.append([])

        #for each row of data
        all_data = [row[i] for row in sample_list]
        for ind in range(len(all_data)):
            class_data[int(class_list[ind])].append(all_data[ind])
        # Here we calculate the fisher ratio for that column
        # calculate the first lumda sqr
        all_data_mean = np.mean(all_data)
        lumdaTop1 = 0
        for z in range(1, classNum+1):
            class_data_mean = np.mean(class_data[z])
            lumdaTop1 = lumdaTop1 + (((class_data_mean - all_data_mean)**2)*len(class_data[z]))
        lumdaBottom1 = classNum-1
        lumda1 = lumdaTop1/lumdaBottom1

        lumdaTop2_1 = 0
        for n in range(1,classNum+1):
            for j in class_data[n]:
                lumdaTop2_1 = lumdaTop2_1 + (j - all_data_mean)**2

        lumdaTop2_2 = 0
        for p in range(1,classNum+1):
            class_data_mean = stat.mean(class_data[p])
            lumdaTop2_2 = lumdaTop2_2 + (((class_data_mean - all_data_mean) ** 2) * len(class_data[p]))
        lumdaBottom2 = len(all_data) - classNum
        lumda2 = (lumdaTop2_1-lumdaTop2_2)/lumdaBottom2
        fisher_ratio = lumda1/lumda2
        fish_ratio.append(fisher_ratio)
    fish_ratio = np.nan_to_num(fish_ratio, nan=(10 ** -12))
    return fish_ratio

def mean_confidence_interval(data, confidence):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

gaussian_algorithm(2)
