import xlrd
import pandas as pd
import numpy as np
import random
import copy
import statistics as stat
from decimal import Decimal
from statistics import NormalDist

def gaussian_algorithm(classNum=2):
    sample_matrix, class_list = getValFromFile("data/setClass_file.xlsx")
    k = 0
    true_means = []
    null_means = []
    while k < 30:
        half_rand_matrix, half_rand_class_list, half_rand_variable_idx_list = selectHalfRandom(sample_matrix, class_list)
        classNum_list = []
        for i in range(classNum):
            classNum_list.append(i + 1)
        true_dist_classNum = copy.deepcopy(half_rand_class_list)
        null_dist_classNum = []
        for i in range(len(half_rand_class_list)):
            null_dist_classNum.append(random.choice(class_list))
        true_fisherRatio = cal_fish_ratio(half_rand_matrix, true_dist_classNum, classNum)
        null_fisherRatio = cal_fish_ratio(half_rand_matrix, null_dist_classNum, classNum)
        true_mean_fisher_ratio = np.mean(true_fisherRatio)
        null_mean_fisher_ratio = np.mean(null_fisherRatio)
        true_means.append(true_mean_fisher_ratio)
        null_means.append(null_mean_fisher_ratio)
        k = k + 1

    # startNum = st.t.interval(alpha=0.999, df=len(true_means)-1, loc=np.mean(true_means), scale=st.sem(true_means))
    # print(startNum)
    # startNum = startNum[1]
    true_fisher_mean = np.mean(true_means)
    true_fisher_std = np.std(true_means)
    null_fisher_mean = np.mean(null_means)
    null_fisher_std = np.std(null_means)
    startNum = NormalDist(mu=true_fisher_mean,sigma=true_fisher_std).inv_cdf(0.99)
    # endNum = get_intersection(true_fisher_mean,null_fisher_mean,true_fisher_std,null_fisher_std)
    # endNum = endNum[0]
    endNum = NormalDist(mu=null_fisher_mean, sigma=null_fisher_std).inv_cdf(0.05)
    # null_m, null_mMinusH, null_mPlusH = mean_confidence_interval(null_fisherRatio, 0.95)
    x = np.linspace(-5, 9, 10000)
    # plot1 = plt.plot(x, norm.pdf(x, true_fisher_mean, true_fisher_std))
    # plot2 = plt.plot(x, norm.pdf(x, null_fisher_mean, null_fisher_std))
    # plot3 = plt.plot(endNum, norm.pdf(endNum, true_fisher_mean, true_fisher_std), 'o')
    # plot4 = plt.plot(startNum, 0.5, 'o')
    # plt.show()
    return startNum,endNum



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

    return final_rand_matrix,rand_class_list,rand_variable_list

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

def one_side_confi_ratio(confidence):
    confidence = Decimal(str(confidence))
    half_confi = Decimal('0.5')
    oneside_confidence = (confidence -half_confi)*Decimal(2)
    oneside_confidence = float(oneside_confidence)

    return oneside_confidence

def get_intersection(m1,m2,std1,std2):
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)


    return np.roots([a, b, c])


