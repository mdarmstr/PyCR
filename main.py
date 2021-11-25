import numpy as np
import setClass
import calculateScore
import xlrd
import newScore
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import genStartEndNum2

def main():
    # generate class number for each sample
    classNum = setClass.setClass('data/CasevControl.xlsx')
    # get the start number and the end number
    startNum, endNum = genStartEndNum2.gaussian_algorithm(classNum)
    # create a hash table to take count for the show up times for each variables
    hash_list = [0]*1500
    for i in range(200):
        # getting the selected index
        return_idx = newScore.setNumber(classNum,startNum,endNum)
        for j in return_idx:
            hash_list[j] = hash_list[j]+1
    valid_idx = []
    # calculate the show-up ratio for each variable
    for i in range(len(hash_list)):
        prob = float(hash_list[i])/200.0
        # we are only taking the ratio more than 30%
        if prob > 0.3:
            print(prob)
            valid_idx.append(i)
    genfile(valid_idx, "data/setClass_file.xlsx")

# generate file of variables by the variable index
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
    df.to_excel("data/finalOutput.xlsx", index=False)

# get the list of samples from the original file
def getValFromFile(fileName):
    wb = xlrd.open_workbook(fileName)
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)
    samples = []
    # add all the variables in cluster into the variables list
    for i in range(1, sheet.nrows):
        temp_col1 = []
        for z in range(3, sheet.ncols):
            temp_col1.append(float(sheet.cell_value(i, z)))
        samples.append(temp_col1)
    independent_y = []
    for j in range(1, sheet.nrows):
        independent_y.append(float(sheet.cell_value(j, (sheet.ncols-1))))
    return samples,independent_y

# def calculatePAC(X,Y):
#     X = np.array(X)
#     Y = np.array(Y)
#     # splitting dataset into a training set and test set
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#     # The next step is to do feature scaling of train and test dataset with help of StandardScaler.
#     sc = StandardScaler()
#     x_train = sc.fit_transform(x_train)
#     x_test = sc.transform(x_test)
#     # We are applying the PCA algorithm for two-component and fitting logistic regression to the training set and predict the result.
#     pca = PCA(n_components=2)
#     X_train = pca.fit_transform(x_train)
#     X_test = pca.transform(x_test)
#
#     explained_variance = pca.explained_variance_ratio_
#     print(explained_variance)
#     x_train = sc.fit_transform(x_train)
#
#     x_test = sc.transform(x_test)
#
#     lab_enc = preprocessing.LabelEncoder()
#     training_scores_encoded = lab_enc.fit_transform(y_train)
#     # fitting logistic Regression to training set
#     classifier = LogisticRegression(random_state=0)
#     classifier.fit(x_train, training_scores_encoded)
#     # predicting results
#
#     y_pred = classifier.predict(x_test)
#     print(y_pred)

main()
