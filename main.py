import fisherRatio
import setClass
import calculateScore
import xlrd
import newScore
import pandas as pd


def main():
    # setClass.setClass('data/CasevControl.xlsx')
    fisher_prob = fisherRatio.cal_ratio('data/setClass_file.xlsx',2)
    hash_list = [0]*1500
    for i in range(200):
        print("###########################################     "+str(i))
        return_idx = newScore.setNumber(fisher_prob, 2)
        for j in return_idx:
            hash_list[j] = hash_list[j]+1
    valid_idx = []
    for i in range(len(hash_list)):
        prob = float(hash_list[i])/200.0
        if prob > 0.8:
            valid_idx.append(i)
    genfile(valid_idx, "data/setClass_file.xlsx")


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

main()
