import setClass
import xlrd
import newScore
import pandas as pd
import genStartEndNum2

def main():
    # get the class list
    classList = getValFromFileByCols('coffee_data/classcoffe.xlsx')
    # get the max class number as classNum
    classNum = max(classList)
    # get the variable lust
    valList = getValFromFileByRows('coffee_data/coffe.xlsx')
    # get the start number and the end number
    startNum, endNum = genStartEndNum2.gaussian_algorithm(int(classNum), classList, valList)
    print(startNum)
    print(endNum)
    # create a hash table to take count for the show up times for each variables
    hash_list = [0]*1500
    for i in range(1):
        # getting the selected index
        return_idx = newScore.setNumber(int(classNum), classList, valList, startNum, endNum)
        for j in return_idx:
            hash_list[j] = hash_list[j]+1
    valid_idx = []
    # calculate the show-up ratio for each variable
    for i in range(len(hash_list)):
        prob = float(hash_list[i])/200.0
        # we are only taking the ratio more than 30%
        print(prob)
        if prob > 0.8:
            valid_idx.append(i)
    # genfile(valid_idx, "coffee_data/coffe.xlsx", "coffee_data/classcoffe.xlsx")

# generate file of variables by the variable index
def genfile(indexList, fileName, classFileName):
    wb = xlrd.open_workbook(classFileName)
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
main()
