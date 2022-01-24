import csv
import gen_clust

def main():
    scores = []
    with open('tempCsv.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            temp_row= []
            for c in row:
                temp_row.append(float(c))
            scores.append(temp_row)
    classList = []
    with open('temp_classList.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            temp_c = []
            for c in row:
                temp_c.append(float(c))
            classList.append(temp_c)
    print(classList[0])
    classList = classList[0]

    gen_clust.RunClust(scores,classList,2)

main()