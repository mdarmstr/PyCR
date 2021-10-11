import fisherRatio
import setClass
import setNumber


def main():
    # setClass.setClass('data/CasevControl.xlsx')
    fisher_prob = fisherRatio.cal_ratio('data/setClass_file.xlsx',2)
    # setNumber.setNumber(fisher_prob,2)
main()
