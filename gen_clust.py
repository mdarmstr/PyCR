import xlrd
import ClustResolution
import numpy as np
def RunClust(variable_list,class_list,classNum):
    # To open Workbook
    # wb = xlrd.open_workbook('data/clust.xlsx')
    # # select the first sheet from xlsx file
    # sheet = wb.sheet_by_index(0)
    # # print(sheet.cell_value(0, 0))
    # # print(sheet.nrows)
    # # define 2 arrays
    # clust1 = []
    # clust2 = []
    # # For loop compare with each group [1 or 2] and put different datda in different clust array
    # for i in range(1, sheet.nrows):
    #     if int(sheet.cell_value(i, 1)) == 1:
    #         temp_col1 = []
    #         for z in range(2,sheet.ncols):
    #             temp_col1.append(float(sheet.cell_value(i, z)))
    #         clust1.append(temp_col1)
    #
    #     elif int(sheet.cell_value(i, 1)) == 2:
    #         temp_col2 = []
    #         for k in range(2,sheet.ncols):
    #             temp_col2.append(float(sheet.cell_value(i,k)))
    #         clust2.append(temp_col2)
    clust1 =[]
    clust2 = []
    for i in range(len(class_list)):
        if class_list[i] == 1:
            clust1.append(variable_list[i])
        elif class_list[i] == 2:
            clust2.append(variable_list[i])
    clust1 = np.array(clust1)
    clust1 = clust1.transpose()
    clust2 = np.array(clust2)
    clust2 = clust2.transpose()
    print(clust1)
    print("#############")
    print(clust2)
    # Call function ClustResilution to do further calculation
    return ClustResolution.clustResolution(clust1, clust2)
