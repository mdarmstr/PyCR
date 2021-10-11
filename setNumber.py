import operator
import pandas as pd
import xlrd
import gen_clust
#set the start number and end number
def setNumber(fisherProb,classNum):
    START_NUM = 0.9
    END_NUM = 0.5
    sorted_fisherProb = sorted(fisherProb.items(), key=operator.itemgetter(1),reverse=True)
    # print(sorted_fisherProb)
    startNumList = []
    endNumList = []
    for i in sorted_fisherProb:
        if i[1] > 0.9:
            startNumList.append(i[0])
        if i[1]< 0.9 and i[1]>0.5:
            endNumList.append(i[0])
    genfile(startNumList)
    clust_resolution = gen_clust.RunClust(classNum)
    print(clust_resolution)


def genfile(indexList):
    wb = xlrd.open_workbook("data/setClass_file.xlsx")
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)
    sample_col = sheet.col_values(0)
    df = pd.DataFrame(sample_col[1:], columns=[sample_col[0]])
    class_col =  sheet.col_values(2)
    df.insert(1,class_col[0],class_col[1:],True)

    loc_counter = 2
    for i in indexList:
        tem_col = sheet.col_values(i)
        df.insert(loc_counter,tem_col[0],tem_col[1:],True)
        loc_counter = loc_counter + 1
    df.to_excel("data/clust.xlsx", index=False)




