import numpy as np
import xlrd
import newScore
import pandas as pd
import genStartEndNum2
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats.distributions import chi2
import sys
from scipy.sparse.linalg import svds
from numpy import inf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xlsxwriter

def main(isexternal,howMuchSplit):
    # get the class list
    classList = getValFromFileByCols('test_data/class_mex.xlsx')
    classMatrix = np.array(classList)
    # get the max class number as classNum
    classNum = max(classList)
    # get the variable list
    sampleList = getValFromFileByRows('test_data/data_mex.xlsx')
    sampleMatrix = np.array(sampleList)
    ## if there is not enough samples to do the external validation no matter what the user says isexternal will be false
    if len(sampleList) < 50:
        isexternal = False
    if isexternal:
        sampleList, external_validation, classList, external_class = selectRandom(sampleList, classList,howMuchSplit)

    # get the start number and the end number
    startNum, endNum = genStartEndNum2.gaussian_algorithm(int(classNum), classList, sampleList)

    # create a file to save the generate statistical number(accuracy, sensitivity, selectivity)
    class_wb_list = []
    class_ws_list = []
    class_stat_list = []
    for classNum in range(1,int(classNum)+1):
        new_wb = xlsxwriter.Workbook('output/class'+str(classNum)+'_stat_report.xlsx')
        new_ws = new_wb.add_worksheet()
        class_wb_list.append(new_wb)
        class_ws_list.append(new_ws)
        class_stat_list.append([])

    # create a hash table to take count for the show up times for each variables
    hash_list = [0]*1500
    auc_table = []
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
                # we are only taking the ratio more than 30%
                if prob > 0.8:
                    valid_idx.append(i)

        selectedVariables = sample_taining[:, valid_idx]

        # Train the model using the training sets
        clf = svm.SVC(kernel='linear', random_state=0, probability=True)
        clf.fit(selectedVariables, class_training)
        class_pred = clf.predict(sampleList[:, valid_idx])

        classofic_report = classification_report(classList, class_pred)
        report_lines = classofic_report.split('\n')
        report_lines = report_lines[2:]
        for c in range(0,classNum):
            stat_num = report_lines[c].split(' ')
            stat_num = [i for i in stat_num if i != ""]
            class_stat_list[c].append(stat_num[1:])
        # generate the roc curve
        class_pred= clf.predict_proba(sampleList[:, valid_idx])
        class_pred = class_pred[:, 1]
        auc = metrics.roc_auc_score(classList, class_pred)
        auc_table.append(auc)
        color_code = 123
        fpr, tpr, _ = metrics.roc_curve(classList, class_pred, pos_label=2)
        plt.plot(fpr, tpr, color=(color_code/235, 50/235, 168/235))
        color_code -= 10
    plt.savefig('output/roc_20_iteration.png')
    plt.figure().clear()
    ## save all the auc number in to list
    auc_row = 0
    auc_wb = xlsxwriter.Workbook('output/auc_report.xlsx')
    auc_ws = auc_wb.add_worksheet()
    for auc_num in auc_table:
        auc_ws.write(auc_row, 0, auc_num)
        auc_row = auc_row+1
    auc_wb.close()
    genfile(valid_idx, 'test_data/data_mex.xlsx')
    row = 0
    # create a xlsx table for the auc numbers
    for c in range(classNum):
        class_ws_list[c].write(row, 0, 'selectivity')
        class_ws_list[c].write(row, 1, 'sensitivity')
        class_ws_list[c].write(row, 2, 'accuracy')

    for c in range(classNum):
        row = 1
        for stat in class_stat_list[c]:
            class_ws_list[c].write(row,0,stat[0])
            class_ws_list[c].write(row,1,stat[1])
            class_ws_list[c].write(row,2,stat[2])
            row = row +1

    for wb in class_wb_list:
        wb.close()

    valid_idx = []
    # calculate the show-up ratio for each variable
    for i in range(len(hash_list)):
        prob = float(hash_list[i])/10
        # we are only taking the ratio more than 30%
        if prob > 0.8:
            valid_idx.append(i)
    ####################################  START GRAPH CODE ###################################
    # generate PCA visualization
    scale_training_sample,scale_training_mean,scale_training_std = scale_half_data(sampleList)
    dummyU, dummyS, V = svds(scale_training_sample[:,valid_idx], k=2)
    V = np.transpose(V)
    Xt = np.dot(scale_training_sample[:,valid_idx], V)
    plot = plt.scatter(Xt[:, 0], Xt[:, 1], c=classList, marker='P')
    class_label_list = []
    for classLabel in range(1, int(classNum) + 1):
        class_label_list.append('training '+ str(classLabel))
    plt.legend(handles=plot.legend_elements()[0], labels=class_label_list)

    if isexternal:
        scaled_external = scale_all_data(external_validation,scale_training_mean,scale_training_std)
        dummyU, dummyS, V = svds(scale_training_sample[:, valid_idx], k=2)
        V = np.transpose(V)
        Xt_external = np.dot(scaled_external[:,valid_idx], V)
        plot = plt.scatter(Xt_external[:, 0], Xt_external[:, 1], c=external_class, marker="o")
        external_class_label_list = []
        for classLabel in range(1, int(classNum) + 1):
            external_class_label_list.append('external ' + str(classLabel))
        plt.legend(handles=plot.legend_elements()[0], labels=external_class_label_list)
    plt.savefig('output/pca.png')
    plt.figure().clear()
    # genfile(valid_idx, "coffee_data/coffe.xlsx")
    # generate ROC for external validation and selected variables
    if isexternal:
        # Create a svm Classifier
        clf = svm.SVC(kernel='linear', random_state=0, probability=True)  # Linear Kernel

        # Train the model using the training sets
        clf.fit(sampleList[:, valid_idx], classList)
        # generate the roc curve
        class_pred = clf.predict_proba(external_validation[:, valid_idx])
        class_pred = class_pred[:, 1]
        auc_external = metrics.roc_auc_score(external_class, class_pred)
        fpr, tpr, _ = metrics.roc_curve(external_class, class_pred, pos_label=2)
        plt.plot(fpr, tpr, label=" auc="+str(auc_external))
        plt.legend(loc=4)
        plt.savefig('output/roc_external.png')
        plt.figure().clear()

    # generate 4 SVM graph
    # graph 1: training without feature selection
    dummyU, dummyS, V = svds(scale_training_sample, k=2)
    V = np.transpose(V)
    Xt_training_noFS = np.dot(scale_training_sample, V)
    plot_train_noFS = plt.scatter(Xt_training_noFS[:, 0], Xt_training_noFS[:, 1], c=classList, marker='P')
    plt.legend(handles=plot_train_noFS.legend_elements()[0], labels=class_label_list)
    plt.savefig('output/PCATrainNoFS.png')
    plt.figure().clear()
    # generate predict ROC
    clf_train_noFS = svm.SVC(kernel='linear', random_state=0, probability=True)
    clf_train_noFS.fit(sampleList, classList)
    class_pred = clf_train_noFS.predict_proba(sampleList)
    class_pred = class_pred[:, 1]
    auc_train_noFS = metrics.roc_auc_score(classList, class_pred)
    fpr, tpr, _ = metrics.roc_curve(classList,  class_pred, pos_label=2)
    plt.plot(fpr, tpr,  label=" auc="+str(auc_train_noFS))
    plt.legend(loc=4)
    plt.savefig('output/rocTrainNoFS.png')
    plt.figure().clear()

    # graph 2: validation without feature selection

    pdummyU, dummyS, V = svds(scale_training_sample, k=2)
    V = np.transpose(V)
    Xt_vali_noFS = np.dot(scaled_external, V)
    plot_vali_noFS = plt.scatter(Xt_vali_noFS[:, 0], Xt_vali_noFS[:, 1], c=external_class, marker='P')
    plt.legend(handles=plot_vali_noFS.legend_elements()[0], labels=class_label_list)
    plt.savefig('output/PCAValiNoFS.png')
    plt.figure().clear()
    # generate predict ROC
    class_pred = clf_train_noFS.predict_proba(external_validation)
    class_pred = class_pred[:, 1]
    auc_vali_noFS = metrics.roc_auc_score(external_class, class_pred)
    fpr, tpr, _ = metrics.roc_curve(external_class, class_pred, pos_label=2)
    plt.plot(fpr, tpr, label=" auc="+str(auc_vali_noFS))
    plt.legend(loc=4)
    plt.savefig('output/rocValiNoFS.png')
    plt.figure().clear()

    # graph 3: training with feature selection
    pdummyU, dummyS, V = svds(scale_training_sample[:, valid_idx], k=2)
    V = np.transpose(V)
    Xt_training_FS = np.dot(scale_training_sample[:, valid_idx], V)
    plot_train_FS = plt.scatter(Xt_training_FS[:, 0], Xt_training_FS[:, 1], c=classList, marker='P')
    plt.legend(handles=plot_train_FS.legend_elements()[0], labels=class_label_list)
    plt.savefig('output/PCATrainWithFS.png')
    plt.figure().clear()
    # generate predict ROC
    clf_train_FS = svm.SVC(kernel='linear', random_state=0, probability=True)
    clf_train_FS.fit(sampleList[:, valid_idx], classList)
    class_pred = clf_train_FS.predict_proba(sampleList[:, valid_idx])
    class_pred = class_pred[:, 1]
    auc_train_FS = metrics.roc_auc_score(classList, class_pred)
    fpr, tpr, _ = metrics.roc_curve(classList, class_pred, pos_label=2)
    plt.plot(fpr, tpr,  label=" auc="+str(auc_train_FS))
    plt.legend(loc=4)
    plt.savefig('output/rocTrainFS.png')
    plt.figure().clear()

    # graph 4: validation with feature selection
    pdummyU, dummyS, V = svds(scale_training_sample[:, valid_idx], k=2)
    V = np.transpose(V)
    Xt_vali_FS = np.dot(scaled_external[:, valid_idx], V)
    plot_vali_FS = plt.scatter(Xt_vali_FS[:, 0], Xt_vali_FS[:, 1], c=external_class, marker='P')
    plt.legend(handles=plot_vali_FS.legend_elements()[0], labels=class_label_list)
    plt.savefig('output/PCAValiWithFS.png')
    plt.figure().clear()
    # generate predict ROC
    class_pred = clf_train_FS.predict_proba(external_validation[:, valid_idx])
    class_pred = class_pred[:, 1]
    auc_vali_FS = metrics.roc_auc_score(external_class, class_pred)
    fpr, tpr, _ = metrics.roc_curve(external_class, class_pred, pos_label=2)
    plt.plot(fpr, tpr, label=" auc="+str(auc_vali_FS))
    plt.legend(loc=4)
    plt.savefig('output/rocValiFS.png')
    plt.figure().clear()
    ####################################  END GRAPH CODE ###################################

# generate file of variables by the variable index
def genfile(indexList, fileName):
    wb = xlrd.open_workbook(fileName)
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)

    first_col = sheet.col_values(indexList[0])
    df = pd.DataFrame(first_col)
    for i in range(1, len(indexList)):
        col = sheet.col_values(indexList[i])
        new_df = pd.DataFrame(col)
        df = pd.concat([df, new_df], axis=1)
    df.to_excel("output/selectVariable.xlsx", index=False,header=None)

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
    return scaled_samples, samples_mean, samples_std

def scale_all_data(samples,mean,std):
    functionTop = np.subtract(samples, mean)
    scaled_samples = np.divide(functionTop, std)
    for list in scaled_samples:
        list[list==inf] = 10**-12
    scaled_samples = np.nan_to_num(scaled_samples, nan=(10**-12))
    return scaled_samples

def selectRandom(sample_list,class_list,howMuchSplit):
    sample_matrix = np.array(sample_list)
    class_matrix = np.array(class_list)
    X_train, X_test, y_train, y_test = train_test_split(sample_matrix, class_matrix, test_size=float(howMuchSplit))
    return X_train, X_test, y_train, y_test

def confident_ellipse(score1, score2, confident_interval = 0.95):


main(True,0.5)
