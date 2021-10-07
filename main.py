import Fisher_ratio
import setClass
def main():
    setClass.setClass('data/CasevControl.xlsx')
    Fisher_ratio.cal_ratio('data/setClass_file.xlsx')
main()
