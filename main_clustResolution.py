import xlrd
import ClustResolution
def main():
    # To open Workbook
    wb = xlrd.open_workbook('data/cluster.xlsx')
    # select the first sheet from xlsx file
    sheet = wb.sheet_by_index(0)
    # print(sheet.cell_value(0, 0))
    # print(sheet.nrows)
    # define 2 arrays
    clust1 = []
    clust2 = []
    # For loop compare with each group [1 or 2] and put different datda in different clust array

    for i in range(1, sheet.nrows):
        if int(sheet.cell_value(i, 2)) == 1:
            clust1.append([float(sheet.cell_value(i, 0)), float(sheet.cell_value(i, 1))])

        elif int(sheet.cell_value(i, 2)) == 2:
            clust2.append([float(sheet.cell_value(i, 0)), float(sheet.cell_value(i, 1))])
    # Call function ClustResilution to do further calculation
    ClustResolution.clustResolution(clust1, clust2)
main()
