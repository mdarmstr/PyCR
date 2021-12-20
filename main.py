import numpy as np
import xlrd
import newScore
import pandas as pd
import genStartEndNum2
from sklearn import svm
import matplotlib.pyplot as plts
from sklearn import metrics
from sklearn.decomposition import PCA
import sys
from numpy import inf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main(isexternal,howMuchSplit):
    # get the class list
    classList = getValFromFileByRows('test_data/class_1.xlsx')[0]
    classMatrix = np.array(classList)
    # get the max class number as classNum
    classNum = max(classList)
    # get the variable list
    sampleList = getValFromFileByRows('test_data/test_1.xlsx')
    sampleMatrix = np.array(sampleList)
    ## is there is not enough samples to do the external validation no matter what the user says isexternal will be false
    if len(sampleList) < 50:
        isexternal = False
    if isexternal:
        sampleList, external_validation, classList, external_class = selectRandom(sampleList, classList,howMuchSplit)

    # get the start number and the end number
    startNum, endNum = genStartEndNum2.gaussian_algorithm(int(classNum), classList, sampleList)

    # create a hash table to take count for the show up times for each variables
    hash_list = [0]*1500

    # Create a svm Classifier -> for SVM graph
    clf = svm.SVC(kernel='linear', random_state=0)  # Linear Kernel
    for k in range(10):
        printStr = "###################################" + str(k)
        print(printStr)
        # getting the selected index
        return_idx, sample_taining, sample_test, class_training, class_test = newScore.setNumber(int(classNum), classList, sampleList, startNum, endNum, howMuchSplit)
        for j in return_idx:
            hash_list[j] = hash_list[j]+1

        if k == 0:
            valid_idx = return_idx
        else:
            valid_idx = []
            # calculate the show-up ratio for each variable
            for i in range(len(hash_list)):
                prob = float(hash_list[i]) /float(k+1)
                print(prob)
                # we are only taking the ratio more than 30%
                if prob > 0.9:
                    valid_idx.append(i)

        selectedVariables = sample_taining[:, valid_idx]

        # Train the model using the training sets
        clf.fit(selectedVariables, class_training)
        # generate the roc curve
        metrics.plot_roc_curve(clf, sample_test[:, valid_idx], class_test)
        # get statistical number

    plt.savefig('imgs/svm200.png')

    valid_idx = []
    # calculate the show-up ratio for each variable
    for i in range(len(hash_list)):
        prob = float(hash_list[i])/10
        # we are only taking the ratio more than 30%
        print(prob)
        if prob > 0.9:
            valid_idx.append(i)
    ####################################  START GRAPH CODE ###################################
    # generate PCA visualization
    scale_training_sample = scale_half_data(sampleList)
    pca = PCA()
    Xt = pca.fit_transform(scale_training_sample[:, valid_idx])
    plot = plt.scatter(Xt[:, 0], Xt[:, 1], c=classList, marker='P')
    class_label_list = []
    for classLabel in range(1, int(classNum) + 1):
        class_label_list.append('training '+ str(classLabel))
    plt.legend(handles=plot.legend_elements()[0], labels=class_label_list)

    if isexternal:
        scale_external_sample = scale_half_data(external_validation)
        Xe = pca.fit_transform(scale_external_sample[:, valid_idx])
        plot = plt.scatter(Xe[:, 0], Xe[:, 1], c=external_class, marker="o")
        external_class_label_list = []
        for classLabel in range(1, int(classNum) + 1):
            external_class_label_list.append('external ' + str(classLabel))
        plt.legend(handles=plot.legend_elements()[0], labels=external_class_label_list)
    plt.savefig('imgs/pca.png')
    genfile(valid_idx, "coffee_data/coffe.xlsx")
    # generate ROC for external validation and selected variables
    if isexternal:
        # Create a svm Classifier
        clf = svm.SVC(kernel='linear', random_state=0)  # Linear Kernel

        # Train the model using the training sets
        clf.fit(sampleList[:,valid_idx], classList)

        # generate the roc curve
        metrics.plot_roc_curve(clf, external_validation[:, valid_idx], external_class)
        plt.savefig('imgs/svm_external.png')
    # generate 4 SVM graph
    # graph 1: training SVM without feature selection
    clf_train_noFS = svm.SVC(kernel='linear', random_state=0)
    clf_train_noFS.fit(sampleList,classList)
    pca_train_noFS = PCA()
    Xt = pca_train_noFS.fit_transform(sampleList)
    plot_train_noFS = plt.scatter(Xt[:, 0], Xt[:, 1], c=classList, marker='P')
    plt.legend(handles=plot_train_noFS.legend_elements()[0], labels=class_label_list)
    plt.savefig('imgs/SVMTrainNoFS.png')
    plt.figure().clear()
    # graph 2: validation SVM without feature selection
    pca_vali_noFS = PCA()
    Xt = pca_vali_noFS.fit_transform(external_validation)
    plot_vali_noFS = plt.scatter(Xt[:, 0], Xt[:, 1], c=external_class, marker='P')
    plt.legend(handles=plot_vali_noFS.legend_elements()[0], labels=class_label_list)
    plt.savefig('imgs/SVMValiNoFS.png')
    plt.figure().clear()
    # graph 2: training SVM with feature selection
    pca_train_FS = PCA()
    Xt = pca_train_FS.fit_transform(sampleList[:, valid_idx])
    plot_train_FS = plt.scatter(Xt[:, 0], Xt[:, 1], c=classList, marker='P')
    plt.legend(handles=plot_train_FS.legend_elements()[0], labels=class_label_list)
    plt.savefig('imgs/SVMTrainWithFS.png')
    plt.figure().clear()
    # graph 2: validation SVM with feature selection
    pca_vali_FS = PCA()
    Xt = pca_vali_FS.fit_transform(external_validation[:, valid_idx])
    plot_vali_FS = plt.scatter(Xt[:, 0], Xt[:, 1], c=external_class, marker='P')
    plt.legend(handles=plot_vali_FS.legend_elements()[0], labels=class_label_list)
    plt.savefig('imgs/SVMValiWithFS.png')
    plt.figure().clear()
    ####################################  END GRAPH CODE ###################################

# generate file of variables by the variable index
def genfile(indexList, fileName):
    wb = xlrd.open_workbook(fileName)
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)

    first_col = sheet.col_values(indexList[0])
    df = pd.DataFrame(first_col)
    for i in range(1,len(indexList)):
        col = sheet.col_values(indexList[i])
        new_df = pd.DataFrame(col)
        df = pd.concat([df,new_df],axis=1)

    df.to_excel("coffee_data/finalOutput.xlsx", index=False,header=None)

# get the list of samples from the original file
def getValFromFileByRows(fileName):
    wb = xlrd.open_workbook(fileName)
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)
    samples = []
    # add all the variables in cluster into the variables list
    for i in range(0, sheet.nrows):
        temp_col1 = []
        for z in range(0, sheet.ncols):
            temp_col1.append(float(sheet.cell_value(i, z)))
        samples.append(temp_col1)
    return samples

def getValFromFileByCols(fileName):
    wb = xlrd.open_workbook(fileName)
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)
    samples = []
    # add all the variables in cluster into the variables list
    for z in range(0, sheet.ncols):
        temp_col1 = []
        for i in range(0, sheet.nrows):
            temp_col1.append(float(sheet.cell_value(i, z)))
        samples.append(temp_col1)
    if sheet.ncols == 1:
        return samples[0]
    else:
        return samples

def scale_half_data(samples):
    # after get all the selected variables we make them a metrix and calculate the mean
    samples = np.array(samples)
    samples_mean = samples.mean(axis=0)
    samples_std = np.std(samples, axis=0)
    np.set_printoptions(threshold=sys.maxsize)
    functionTop = np.subtract(samples, samples_mean)
    scaled_samples = np.divide(functionTop, samples_std)
    scaled_samples = np.nan_to_num(scaled_samples, nan=(10**-12))
    for list in scaled_samples:
        list[list==inf] = 10**-12

    return scaled_samples

def selectRandom(sample_list,class_list,howMuchSplit):
    sample_matrix = np.array(sample_list)
    class_matrix = np.array(class_list)
    X_train, X_test, y_train, y_test = train_test_split(sample_matrix, class_matrix, test_size=float(howMuchSplit))
    return X_train, X_test, y_train, y_test

main(True,0.5)
