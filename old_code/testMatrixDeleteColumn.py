import numpy
import numpy as np
import itertools
import math
from scipy import stats
def main():
    # list_num = [[1,2,3],[4,5,6],[7,8,9]]
    # list_num = np.array(list_num)
    # print(list_num)
    # deleted_list = np.delete(list_num,[0,2],1)
    # print(deleted_list)
    # inserted_list = numpy.insert(deleted_list,0,[7,8,9],axis=1)
    # print(inserted_list)
    # print(inserted_list[:,1])
    # inserted_list = numpy.insert(inserted_list, 0, [5, 6, 7], axis=1)
    # inserted_list = numpy.insert(inserted_list, 0, [5, 8, 8], axis=1)
    # print(inserted_list)
    # X = inserted_list[:, [0, 1, 3]]
    # print("X")
    # print(X)
    # list2 = [1,2,3,4,5,6]
    # list2 = np.array(list2)
    # print(list2)
    # print(list2+1)
    # list2 = list(list2)
    # list2.append(9)
    # print(list2)
    # print(list2[:2])
    # print(math.comb(4,2))
    # list_combi = np.array(list(itertools.combinations([1,2,3,4], 2)))
    # print(list_combi)

    # list_num = [[1,2,3],[4,5,6],[7,8,9]]
    # list_num = np.array(list_num)
    # print(list_num)
    # X = list_num[:,[0]]
    # print(X)

    an_array = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]
    an_array = np.array(an_array)
    column_one = an_array[:,[2,1]]
    print(column_one)






main()