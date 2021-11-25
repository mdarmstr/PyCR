import xlrd
import statistics as stat
import column_class
from scipy import stats
import pandas as pd
def cal_ratio(fileName,classNum):
    wb = xlrd.open_workbook(fileName)
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)

    # get the total number of column and row number
    colNum = sheet.ncols
    rowNum = sheet.nrows


    # define a fisher ratio list for all columns with default value 0
    fisherProbDic = {}
    colObjectList = []
    fish_ratio = []
    fish_ratio.append("FISHER RATIO")
    fish_ratio.append(0)
    fish_ratio.append(0)
    # for each column sample type we calculate one fisher ratio for one column
    for i in range(3, colNum):
        #define a data list for all class
        all_data = []
        columnObject = column_class.column(i)
        # define a data list contain different class data list
        class_data = []
        for k in range(classNum + 1):
            class_data.append([])

        #for each row of data
        for j in range(1, rowNum):
            val = sheet.cell_value(j, i)
            class_num = int(sheet.cell_value(j, 2))
            all_data.append(val)
            class_data[class_num].append(val)
        # Here we calculate the fisher ratio for that column
        # calculate the first lumda sqr
        all_data_mean = stat.mean(all_data)
        lumdaTop1 = 0
        for z in range(1, classNum+1):
            class_data_mean = stat.mean(class_data[z])
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
        columnObject.setFisherRatio(fisher_ratio)
        colObjectList.append(columnObject)
        fisher_prob = stats.f.cdf(fisher_ratio, classNum-1, len(all_data)-classNum)

        # print("########################")
        # print(fisher_prob)
        fisherProbDic[i] = fisher_ratio
    # temp_df = pd.read_excel("data/setClass_file.xlsx")
    # temp_df.loc[63] = fish_ratio
    # temp_df.to_excel("data/output.xlsx", index=False)

    return fisherProbDic