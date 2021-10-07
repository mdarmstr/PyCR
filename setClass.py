import xlrd
def setClass(fileName):
    wb = xlrd.open_workbook('data/CasevControl.xlsx')
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)
    #define a class dictionary for later use
    class_dict = {}
    class_value_counter = 1
    # save the first class item into our class dictionary
    class_dict[sheet.cell_value(1, 1)] = class_value_counter
    class_value_counter = class_value_counter + 1
    # check all the item and add new class into class dictionary
    for i in range(1, sheet.nrows):
        if sheet.cell_value(i,1) in class_dict.keys():
            pass
        else:
            class_dict[sheet.cell_value(i, 1)] = class_value_counter
            class_value_counter = class_value_counter + 1
