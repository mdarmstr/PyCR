import ClustResolution
import numpy as np
import itertools
import xlrd

def RunClust():
    variable_list = getValFromFileByRows("dubug_testScore/test1_data.xlsx")
    class_list = getValFromFileByCols("dubug_testScore/test1_class.xlsx")
    classNum = int(max(class_list))
    if classNum == 2:
        clust1 =[]
        clust2 = []
        for i in range(len(class_list)):
            if class_list[i] == 1:
                clust1.append(variable_list[i])
            elif class_list[i] == 2:
                clust2.append(variable_list[i])
        clust1 = np.array(clust1)
        clust1 = clust1
        clust2 = np.array(clust2)
        clust2 = clust2
        # Call function ClustResilution to do further calculation
        print("result of clust resolution")
        print(ClustResolution.clustResolution(clust1, clust2))
        return
    else:
        classNumList = []
        # give you the all labels in a list classNUm = 3, labels = [1,2,3]
        for i in range(classNum):
            classNumList.append(i+1)
        list_combi = np.array(list(itertools.combinations(classNumList, 2)))
        # gives up [[1,2][1,3][2,3]]
        outputClust = 1
        set_cr = {}
        for set in list_combi:
            clust1 = []
            clust2 = []
            for i in range(len(class_list)):
                if class_list[i] == set[0]:
                    clust1.append(variable_list[i])
                elif class_list[i] == set[1]:
                    clust2.append(variable_list[i])
            clust1 = np.array(clust1)
            clust1 = clust1
            clust2 = np.array(clust2)
            clust2 = clust2
            # Call function ClustResilution to do further calculation
            newClust = ClustResolution.clustResolution(clust1, clust2)
            set_cr[tuple(set)] = newClust
            outputClust *=newClust
        for key in set_cr.keys():
            print("set :"+str(key))
            print("cr :" + str(set_cr[key]))
        return

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

RunClust()
