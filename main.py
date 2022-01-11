import numpy as np
import xlrd
import newScore
import pandas as pd
import genStartEndNum2
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from scipy.stats.distributions import chi2
from sklearn.multiclass import OneVsRestClassifier
import sys
from scipy.sparse.linalg import svds
from numpy import inf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xlsxwriter
import math
from colour import Color
from sklearn.preprocessing import label_binarize

def main(isexternal,howMuchSplit,isMicro):
    iteration = 100
    inputDataFileName = 'test_data/data_Mixoils_dataset.xlsx'
    inputClassFileName = 'test_data/class_Mixoils_dataset.xlsx'
    # get the class list
    classList = getValFromFileByCols(inputClassFileName)
    classList = [int(x[0]) for x in classList]
    classMatrix = np.array(classList)
    # generate roc color
    red = Color("#dc3c40")
    roc_colors = list(red.range_to(Color("#55a6bc"), iteration+1))
    # get the max class number as classNum
    classNum = len(np.unique(classMatrix))
    real_class_num = np.unique(classMatrix)
    #trans the class
    class_trans_dict = {}
    for i in range(len(real_class_num)):
        class_trans_dict[i+1] = real_class_num[i]
    for key in class_trans_dict.keys():
        classList = np.where(classList == class_trans_dict.get(key),key,classList)
    classList =  classList.tolist()
    class_num_label = []
    for i in range(1,len(real_class_num)+1):
        class_num_label.append(i)

    #class color
    class_color = ["#dc3c40", "#55a6bc", 'purple', 'yellowgreen', 'wheat', 'royalblue','#42d7f5','#ca7cf7','#d2f77c']
    class_label = ["o", "x", "4", "*", "+", "D", "8", "s", "p"]
    # get the variable list
    sampleList = getValFromFileByRows(inputDataFileName)
    sampleMatrix = np.array(sampleList)
    hori_index = np.arange(1, len(sampleList[0])+1)
    indice_list = np.arange(1, len(classList) + 1)
    export_file(sampleList, classList, indice_list, hori_index, 'output/original_file.xlsx', class_trans_dict)
    ## if there is not enough samples to do the external validation no matter what the user says isexternal will be false
    if len(sampleList) < 50:
        isexternal = False
    ## use hash table to see how many samples for each class and if countSample < 9 we dont do external
    hash_classCount = [0]*(classNum+1)
    for c_num in classList:
        hash_classCount[c_num] += 1
    for i in range(1,classNum+1):
        if hash_classCount[i] < 9:
            isexternal = False
    if isexternal:
        sampleList, external_validation, classList, external_class, indices_train, indices_test = selectRandom(sampleList, classList, howMuchSplit)

    # output the splited training and external variables in special format
    if isexternal:
        export_file(sampleList, classList, indices_train, hori_index, 'output/training_variables.xlsx', class_trans_dict)
        export_file(external_validation, external_class, indices_test, hori_index, 'output/external_variables.xlsx', class_trans_dict)
    else:
        export_file(sampleList, classList, indice_list, hori_index, 'output/training_variables.xlsx', class_trans_dict)
        external_variables_wb = xlsxwriter.Workbook('output/external_variables.xlsx')
        external_variables_ws = external_variables_wb.add_worksheet()
        external_variables_ws.write(0, 0, "There is not enough samples to have external validation.")

    # get the start number and the end number
    startNum, endNum = genStartEndNum2.gaussian_algorithm(int(classNum), classList, sampleList,10)
    # create a file to save the generate statistical number(accuracy, sensitivity, selectivity)
    class_ws_list = []
    class_stat_list = []
    training_iteration_wb = xlsxwriter.Workbook('output/training_stat_report.xlsx')
    for classNum in range(1, int(classNum)+1):
        new_ws = training_iteration_wb.add_worksheet("class " + str(classNum))
        class_ws_list.append(new_ws)
        class_stat_list.append([])

    # create a hash table to take count for the show up times for each variables
    hash_list = [0]*1500
    if classNum == 2 or isMicro:
        auc_table = []
    else:
        fig, axarr = plt.subplots(classNum)
        auc_table = []
        for i in range(classNum):
            auc_table.append([])

    for k in range(iteration):
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
                prob = float(hash_list[i])/float(k+1)
                # we are only taking the ratio more than 30%
                if prob > 0.95:
                    valid_idx.append(i)

        selectedVariables = sample_taining[:, valid_idx]
        # Train the model using the training sets
        clf = svm.SVC(kernel='linear', random_state=0, probability=True)
        clf.fit(selectedVariables, class_training)
        class_pred = clf.predict(sample_test[:, valid_idx])
        classofic_report = classification_report(class_test, class_pred)
        report_lines = classofic_report.split('\n')
        report_lines = report_lines[2:]
        for c in range(0,classNum):
            stat_num = report_lines[c].split(' ')
            stat_num = [i for i in stat_num if i != ""]
            class_stat_list[c].append(stat_num[1:])
        # generate the roc curve
        if classNum == 2:
            class_pred = clf.predict_proba(sample_test[:, valid_idx])
            class_pred = class_pred[:, 1]
            auc_num = metrics.roc_auc_score(class_test, class_pred)
            auc_table.append(auc_num)
            fpr, tpr, _ = metrics.roc_curve(class_test, class_pred, pos_label=2)
            plt.plot(fpr, tpr, color=str(roc_colors[k]))
            auc_table.append(auc_num)
            plt.rcParams.update({'font.size': 21})
            plt.title('Roc_' + str(iteration) + '_iterations')
        else:
            training_class = label_binarize(class_training, classes=class_num_label)
            predict_class = label_binarize(class_test, classes=class_num_label)
            classifier = OneVsRestClassifier(
                svm.SVC(kernel="linear", probability=True, random_state=0)
            )
            y_score = classifier.fit(selectedVariables, training_class).decision_function(sample_test[:,valid_idx])

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(classNum):
                fpr[i], tpr[i], _ = metrics.roc_curve(predict_class[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(predict_class.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            if isMicro:
                plt.plot(
                    fpr["micro"],
                    tpr["micro"],
                    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
                    color=str(roc_colors[k]),
                )
                auc_table.append(roc_auc["micro"])
                plt.rcParams.update({'font.size': 21})
                plt.title('Roc_' + str(iteration) + '_iterations')
            else:

                for i in range(classNum):
                    auc_table[i].append(roc_auc[i])
                    axarr[i].plot(
                        fpr[i],
                        tpr[i],
                        color=str(roc_colors[k]),
                    )
                plt.rcParams.update({'font.size': 21})
                axarr[0].set_title('Roc_' + str(iteration) + '_iterations')

            # fpr["micro"], tpr["micro"], _ = metrics.roc_curve(predict_class.ravel(), y_score.ravel())
            # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.rcParams.update({'font.size': 21})

    plt.savefig('output/roc_'+str(k)+'iterations.png')
    plt.figure().clear()

    ## save all the auc number in to list
    auc_wb = xlsxwriter.Workbook('output/auc_report.xlsx')
    if classNum ==2 or isMicro:
        auc_row = 0
        auc_ws = auc_wb.add_worksheet()
        for auc_num in auc_table:
            auc_ws.write(auc_row, 0, auc_num)
            auc_row = auc_row+1
    else:
        for i in range(classNum):
            temp_ws = auc_wb.add_worksheet('class '+ str(i+1))
            auc_row = 0
            for auc_num in auc_table[i]:
                temp_ws.write(auc_row, 0, auc_num)
                auc_row = auc_row + 1

    auc_wb.close()
    row = 0
    # generate file for selected training and selected validation in special format
    export_vali_index = [x+1 for x in valid_idx]
    if isexternal:
        export_file(sampleList[:,valid_idx], classList, indices_train, export_vali_index, 'output/selected_training_variables.xlsx', class_trans_dict)
        export_file(external_validation[:,valid_idx], external_class, indices_test, export_vali_index, 'output/selected_external_variables.xlsx', class_trans_dict)
    else:
        export_file(sampleList[:,valid_idx], classList,indice_list,export_vali_index, 'output/selected_training_variables.xlsx', class_trans_dict)
        external_variables_wb = xlsxwriter.Workbook('output/selected_external_variables.xlsx')
        external_variables_ws = external_variables_wb.add_worksheet()
        external_variables_ws.write(0, 0, "There is not enough samples to have external validation.")

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

    training_iteration_wb.close()

    valid_idx = []
    # calculate the show-up ratio for each variable
    for i in range(len(hash_list)):
        prob = float(hash_list[i])/iteration
        # we are only taking the ratio more than 30%
        if prob > 0.95:
            valid_idx.append(i)
    ####################################  START GRAPH CODE ###################################
    # generate PCA visualization
    scale_training_sample,scale_training_mean,scale_training_std = scale_half_data(sampleList)
    scaled_external,scale_training_mean,scale_training_std = scale_half_data(external_validation)
    # scaled_external = scale_all_data(external_validation, scale_training_mean, scale_training_std)
    class_index_list = []
    external_class_index_list = []
    for i in range(classNum+1):
        class_index_list.append([])
    for i in range(len(classList)):
        class_index_list[classList[i]].append(i)

    for i in range(classNum+1):
        external_class_index_list.append([])
    for i in range(len(classList)):
        external_class_index_list[external_class[i]].append(i)

    class_variables = scale_training_sample[:, valid_idx]
    dummyU, dummyS, V = svds(class_variables, k=2)
    dummyS = np.diag(dummyS)
    score = np.dot(dummyU, dummyS)
    V = np.transpose(V)
    Xt = np.dot(dummyU, dummyS)

    for z in range(1, classNum+1):
        class_score = score[class_index_list[z],:]
        x_ellipse, y_ellipse = confident_ellipse(class_score[:, 0], class_score[:, 1])
        plt.plot(x_ellipse, y_ellipse,color=class_color[z-1])
        plt.fill(x_ellipse, y_ellipse,color=class_color[z-1], alpha=0.3)
        class_Xt = Xt[class_index_list[z], :]
        plt.scatter(class_Xt[:, 0], class_Xt[:, 1], c=class_color[z-1], marker=class_label[0], label='training' + str(z))
    # calculating the PCA percentage value
    pU, pS, pV = np.linalg.svd(class_variables)
    pca_percentage_val = np.cumsum(pS) / sum(pS)
    p1_percentage = pca_percentage_val[0] * 100
    p2_percentage = pca_percentage_val[1] * 100
    plt.xlabel("P1 \n P1: = {0:0.3f}".format(p1_percentage)+"%")
    plt.ylabel("P2 \n P2: = {0:0.3f}".format(p2_percentage)+"%")
    plt.rcParams.update({'font.size': 21})
    plt.title('PCA_training')
    plt.legend()
    plt.savefig('output/pca_taining.png')
    if isexternal:
        external_Xt = np.dot(scaled_external[:,valid_idx], V)
        for n in range(1, classNum+1):
            class_external_Xt = external_Xt[external_class_index_list[n], :]
            plt.scatter(class_external_Xt[:, 0], class_external_Xt[:, 1], c=class_color[n-1], marker=class_label[1],
                               label='external' + str(n))
        clf_extern = svm.SVC(kernel='linear', random_state=0, probability=True)
        clf_extern.fit(sampleList[:,valid_idx], classList)
        class_pred = clf_extern.predict(external_validation[:, valid_idx])
        classofic_report = classification_report(external_class, class_pred)
        plt.title('PCA_Training_VS_Validation')
        plt.rcParams.update({'font.size': 21})
        plt.legend()
        plt.savefig('output/pca_external.png')
        plt.figure().clear()
        report_lines = classofic_report.split('\n')
        report_lines = report_lines[2:]
        external_stat_wb = xlsxwriter.Workbook('output/external_stat_report_class.xlsx')
        for c in range(0, classNum):
            temp_ws = external_stat_wb.add_worksheet("class " + str(c))
            stat_num = report_lines[c].split(' ')
            stat_num = [i for i in stat_num if i != ""]
            data = stat_num[1:]
            temp_ws.write(0,0,data[0])
            temp_ws.write(0,1,data[0])
            temp_ws.write(0,2,data[0])
        external_stat_wb.close()

    # genfile(valid_idx, "coffee_data/coffe.xlsx")
    # generate ROC for external validation and selected variables
    if isexternal:
        if classNum == 2:
           gen_roc_graph(sampleList[:,valid_idx],classList,external_validation[:,valid_idx],external_class,"output/roc_external.png", 'Roc_External')
        else:
            mul_roc_graph(classNum,class_num_label,classList,external_class,sampleList[:,valid_idx],external_validation[:,valid_idx],roc_colors,"output/roc_external.png",isMicro,'Roc_external')


    # generate 4 SVM graph
    # graph 1: training without feature selection
    gen_pca(scale_training_sample, classNum, class_index_list, class_color, class_label, 'output/PCATrainNoFS.png','PCA_Training_No_FeatureSelection')
    # generate predict ROC
    if classNum == 2:
        gen_roc_graph(sampleList,classList,sampleList,classList,'output/rocTrainNoFS.png','Roc_Training_No_FeatureSelection')
    else:
        mul_roc_graph(classNum, class_num_label, classList, classList, sampleList,
                      sampleList, roc_colors, 'output/rocTrainNoFS.png',isMicro,'Roc_Training_No_FeatureSelection')

    # graph 2: validation without feature selection
    gen_pca(scaled_external, classNum, external_class_index_list, class_color, class_label, 'output/PCAValiNoFS.png','PCA_Validation_No_FeatureSelection')
    # generate predict ROC
    if classNum == 2:
        gen_roc_graph(sampleList,classList,external_validation,external_class,'output/rocValiNoFS.png', 'Roc_Validation_No_FeatureSelection')
    else:
        mul_roc_graph(classNum, class_num_label, classList, external_class, sampleList,
                      external_validation, roc_colors, 'output/rocValiNoFS.png',isMicro,'Roc_Validation_No_FeatureSelection')


    # graph 3: training with feature selection
    gen_pca(scale_training_sample[:,valid_idx], classNum, class_index_list, class_color, class_label,'output/PCATrainWithFS.png','PCA_Training_With_FeatureSelection')
    # generate predict ROC
    if classNum == 2:
        gen_roc_graph(sampleList[:,valid_idx],classList,sampleList[:,valid_idx],classList,'output/rocTrainFS.png','Roc_Training_With_FeatureSelection')
    else:
        mul_roc_graph(classNum, class_num_label, classList, classList, sampleList[:,valid_idx],
                      sampleList[:, valid_idx], roc_colors, 'output/rocTrainFS.png',isMicro, 'Roc_Training_With_FeatureSelection')


    # graph 4: validation with feature selection
    gen_pca(scaled_external[:, valid_idx], classNum, external_class_index_list, class_color, class_label,
            'output/PCAValiWithFS.png','PCA_Validation_With_FeatureSelection')

    # generate predict ROC
    if classNum == 2:
        gen_roc_graph(sampleList[:,valid_idx],classList,external_validation[:,valid_idx],external_class,'output/rocValiFS.png', 'Roc_Validation_With_FeatureSelection' )
    else:
        mul_roc_graph(classNum, class_num_label, classList, external_class, sampleList[:, valid_idx],
                      external_validation[:, valid_idx], roc_colors,'output/rocValiFS.png',isMicro, 'Roc_Validation_With_FeatureSelection' )
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
    indices = np.arange(1,len(class_list)+1)
    sample_matrix = np.array(sample_list)
    class_matrix = np.array(class_list)
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(sample_matrix, class_matrix, indices, test_size=float(howMuchSplit), stratify=class_matrix)
    return X_train, X_test, y_train, y_test, indices_train, indices_test

def confident_ellipse(score1, score2, confident_interval = 0.95):
    score1 = np.array(score1)
    score2 = np.array(score2)
    chi_2 = chi2.ppf(confident_interval, df=2)
    d1 = score1.mean(axis=0)
    d2 = score2.mean(axis=0)
    data = [score1,score2]
    covMat = np.cov(data)
    eivec, eigval, Vh1 = np.linalg.svd(covMat)
    phi1 = math.atan2(eivec[0][1], eivec[0][0])
    if phi1 < 0:
        phi1 = phi1 + 2*math.pi
    theta = np.arange(0, 2*math.pi, 0.01)
    x_ellipse = []
    y_ellipse = []

    for i in theta:
        x_temp = d1 + math.sqrt(chi_2) * math.sqrt(eigval[0]) * math.cos(i) * math.cos(phi1) - math.sqrt(chi_2) * math.sqrt(eigval[1]) * math.sin(i) * math.sin(phi1)
        y_temp = d2 + math.sqrt(chi_2) * math.sqrt(eigval[0]) * math.cos(i) * math.sin(phi1) + math.sqrt(chi_2) * math.sqrt(eigval[1]) * math.sin(i) * math.cos(phi1)
        y_ellipse.append(y_temp)
        x_ellipse.append(x_temp)
    return x_ellipse, y_ellipse

def export_file(variable, class_list, indice, hori, filename, label_dic):

    temp_wb = xlsxwriter.Workbook(filename)
    temp_ws = temp_wb.add_worksheet()
    class_list = np.array(class_list)
    for key in label_dic.keys():
        class_list = np.where(class_list == key, label_dic.get(key), class_list)
    class_list = class_list.tolist()
    ## set the first column
    for row in range(1,len(class_list)+1):
        temp_ws.write(row, 0, "C" + str(indice[row-1]))

    ## set the first row
    temp_ws.write(0, 1, "class")
    temp_ws.write(0, 0, "Sample name")
    for col in range(2,len(variable[0])+2):
        temp_ws.write(0,col,"variable" + str(hori[col-2]))

    ## appen class number
    for row in range(1,len(class_list)+1):
        temp_ws.write(row,1,class_list[row-1])
    ## append variable
    for col in range(2,len(variable[0])+2):
        for row in range(1,len(class_list)+1 ):
            temp_ws.write(row,col,variable[row-1][col-2])
    temp_wb.close()
def mul_roc_graph(classNum, class_num_label, trainingClass, predicClass, trainingVal, predicVal, roc_colors, output_filename,isMicro,graph_title):

    training_class = label_binarize(trainingClass, classes=class_num_label)
    predict_class = label_binarize(predicClass, classes=class_num_label)
    classifier = OneVsRestClassifier(
        svm.SVC(kernel="linear", probability=True, random_state=0)
    )
    y_score = classifier.fit(trainingVal, training_class).decision_function(
        predicVal)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classNum):
        fpr[i], tpr[i], _ = metrics.roc_curve(predict_class[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(predict_class.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    if isMicro:
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color=str(roc_colors[1]),
        )
    else:
        f, axarr = plt.subplots(classNum)
        for i in range(classNum):
            axarr[i].plot(
                fpr[i],
                tpr[i],
                color=str(roc_colors[1]),
                label="ROC curve (area = %0.3f)" % roc_auc[i],
            )
            axarr[i].legend()
    plt.rcParams.update({'font.size': 21})
    axarr[0].set_title(graph_title)
    plt.savefig(output_filename)
    plt.figure().clear()
def gen_roc_graph(training_sample,training_class,predict_sample,predict_class, fileName,graph_title):
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear', random_state=0, probability=True)  # Linear Kernel
    # Train the model using the training sets
    clf.fit(training_sample, training_class)
    class_pred = clf.predict_proba(predict_sample)
    class_pred = class_pred[:, 1]
    auc_external = metrics.roc_auc_score(predict_class, class_pred)
    fpr, tpr, _ = metrics.roc_curve(predict_class, class_pred, pos_label=2)
    plt.plot(fpr, tpr, label="micro-average ROC curve (area = {0:0.3f})".format(auc_external))
    plt.title(graph_title)
    plt.rcParams.update({'font.size': 21})
    plt.legend(loc=4)
    plt.savefig(fileName)
    plt.figure().clear()

def gen_pca(training_sample,classNum,class_index_list,class_color,class_label,fileName,graph_title):
    dummyU, dummyS, V = svds(training_sample, k=2)
    V = np.transpose(V)
    Xt_training_noFS = np.dot(training_sample, V)
    for z in range(1, classNum + 1):
        class_Xt_training_noFS = Xt_training_noFS[class_index_list[z], :]
        plt.scatter(class_Xt_training_noFS[:, 0], class_Xt_training_noFS[:, 1], c=class_color[z - 1],
                    marker=class_label[0], label='class' + str(z))
    # calculating the PCA percentage value
    pU, pS, pV = np.linalg.svd(training_sample)
    pca_percentage_val = np.cumsum(pS) / sum(pS)
    p1_percentage = pca_percentage_val[0] * 100
    p2_percentage = pca_percentage_val[1] * 100
    plt.xlabel("P1 \n P1: = {0:0.3f}".format(p1_percentage) + "%")
    plt.ylabel("P2 \n P2: = {0:0.3f}".format(p2_percentage) + "%")
    plt.title(graph_title)
    plt.rcParams.update({'font.size': 20})
    plt.legend()
    plt.savefig(fileName)
    plt.figure().clear()
main(True,0.5,False)
