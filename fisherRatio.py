import xlrd
import statistics as stat
import column_class
def cal_ratio(fileName,classNum):
    wb = xlrd.open_workbook(fileName)
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)

    # get the total number of column and row number
    colNum = sheet.ncols
    rowNum = sheet.nrows

    # define a fisher ratio list for all columns with default value 0
    Fisher_list = [0] * colNum
    colObjectList = []
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
        for i in range(1, classNum+1):
            class_data_mean = stat.mean(class_data[i])
            lumdaTop1 =  lumdaTop1 + ((class_data_mean - all_data_mean)**2)*i
        lumdaBottom1 = classNum-1
        lumda1 = lumdaTop1/lumdaBottom1

        lumdaTop2_1 = 0
        for i in range(1,classNum+1):
            for j in class_data[i]:
                lumdaTop2_1 = lumdaTop2_1 + (j - all_data_mean)**2

        lumdaTop2_2 = 0
        for i in range(1,classNum+1):
            class_data_mean = stat.mean(class_data[i])
            lumdaTop2_2 = lumdaTop2_2 + ((class_data_mean - all_data_mean) ** 2) * i
        lumdaBottom2 = len(all_data) - classNum
        lumda2 = (lumdaTop2_1-lumdaTop2_2)/lumdaBottom2

        fisher_ratio = lumda1/lumda2
        columnObject.setFisherRatio(fisher_ratio)
        print(fisher_ratio)
        colObjectList.append(columnObject)
    # for i in colObjectList:
    #     print(i.ratio)
    return colObjectList
