import xlrd
import pandas as pd
def setClass(fileName):
    wb = xlrd.open_workbook(fileName)
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)
    #define a class dictionary for later use
    class_dict = {}
    class_value_counter = 1

    # define a class list to add column in our excel file
    classList = []

    # save the first class item into our class dictionary
    class_dict[sheet.cell_value(1, 1)] = class_value_counter
    classList.append(class_value_counter)
    class_value_counter = class_value_counter + 1



    # check all the item and add new class into class dictionary
    for i in range(2, sheet.nrows):
        key = sheet.cell_value(i, 1)
        # first we need to check if the class was registered in the dictionary
        if key in class_dict.keys():
            # if the key already registered, we need to get value and append in to class list
            val = class_dict.get(key)
            classList.append(val)
        else:
            # if not register yet we need to register it and append class number into the class list
            class_dict[key] = class_value_counter
            classList.append(class_value_counter)
            class_value_counter = class_value_counter + 1
    df = pd.read_excel(fileName)
    df.insert(2,"ClassNum",classList,True)
    df.to_excel("data/setClass_file.xlsx", index=False)
    print(classList)

