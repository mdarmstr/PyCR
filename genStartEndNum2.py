import numpy as np
import random
import copy
import statistics as stat
from statistics import NormalDist
import matplotlib.pyplot as plt
from pylab import rcParams

import scipy.stats as st

def gaussian_algorithm(classNum,class_list,valList):
    sample_matrix = np.array(valList)
    k = 0
    true_means = []
    null_means = []
    while k < 30:
        # get the half random sample and half random variables list
        half_rand_matrix,sample_ind_list = selectHalfRandom(sample_matrix)
        half_rand_class_list = []
        for z in sample_ind_list:
            half_rand_class_list.append(class_list[z])
        classNum_list = []
        for i in range(classNum):
            classNum_list.append(i + 1)
        # create true distribution with the original class number
        true_dist_classNum = copy.deepcopy(half_rand_class_list)
        # create null distribution with the randomly assigned class number
        null_dist_classNum = []
        for i in range(len(half_rand_class_list)):
            null_dist_classNum.append(random.choice(classNum_list))
        # calculate the tru and null fisher ratio
        true_fisherRatio = cal_fish_ratio(half_rand_matrix, true_dist_classNum, classNum)
        null_fisherRatio = cal_fish_ratio(half_rand_matrix, null_dist_classNum, classNum)
        true_mean_fisher_ratio = np.mean(true_fisherRatio)
        null_mean_fisher_ratio = np.mean(null_fisherRatio)
        true_means.append(true_mean_fisher_ratio)
        null_means.append(null_mean_fisher_ratio)
        k = k + 1
    ####################################  START GRAPH CODE ###################################
    # generate a histogram by using mean
    rcParams['figure.figsize'] = 10, 10
    plt.hist(true_means, density=True, label="true fisher mean")
    plt.ylabel("Probability")
    plt.xlabel("Mean")
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(true_means)
    plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    plt.hist(null_means, density=True, label=" null fihser mean")
    plt.ylabel("Probability")
    plt.xlabel("Mean")
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(null_means)
    plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    plt.tight_layout()
    plt.savefig('imgs/FisherMean.png')
    ####################################  END GRAPH CODE ###################################

    true_fisher_mean = np.mean(true_means)
    true_fisher_std = np.std(true_means)
    null_fisher_mean = np.mean(null_means)
    null_fisher_std = np.std(null_means)
    startNum = NormalDist(mu=true_fisher_mean,sigma=true_fisher_std).inv_cdf(0.99)
    endNum = NormalDist(mu=null_fisher_mean, sigma=null_fisher_std).inv_cdf(0.05)
    return startNum, endNum


# randomly get half sample, half variable matrix
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