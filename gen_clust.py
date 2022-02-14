import ClustResolution
import numpy as np
import itertools

def RunClust(variable_list,class_list,classNum):
    if classNum == 2:
        clust1 =[]
        clust2 = []
        for i in range(len(class_list)):
            if class_list[i] == 1:
                clust1.append(variable_list[i])
            elif class_list[i] == 2:
                clust2.append(variable_list[i])
        clust1 = np.array(clust1)
        clust2 = np.array(clust2)
        # Call function ClustResilution to do further calculation
        return ClustResolution.clustResolution(clust1, clust2)
    else:
        classNumList = []
        # give you the all labels in a list classNUm = 3, labels = [1,2,3]
        for i in range(classNum):
            classNumList.append(i+1)
        list_combi = np.array(list(itertools.combinations(classNumList, 2)))
        # gives up [[1,2][1,3][2,3]]
        outputClust = 1
        for set in list_combi:
            clust1 = []
            clust2 = []
            for i in range(len(class_list)):
                if class_list[i] == set[0]:
                    clust1.append(variable_list[i])
                elif class_list[i] == set[1]:
                    clust2.append(variable_list[i])
            clust1 = np.array(clust1)
            clust2 = np.array(clust2)
            # Call function ClustResilution to do further calculation
            newClust = ClustResolution.clustResolution(clust1, clust2)
            outputClust *=newClust
        return outputClust



