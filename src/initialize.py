import random

def initial(isBound):
    mapListBound = [[1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2],
               [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2],
               [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2],
                [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2]]

    mapListBlue = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]

    mapListRed = [[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]]


    soldiersA = [7, 8, 4, 10, 9, 8, 7, 6, 5, 7, 6, 7, 10, 9, 7, 10, 9, 8, 3, 5, 4]
    soldiersB = [7, 5, 9, 10, 9, 8, 7, 6, 5, 9, 8, 10, 8, 6, 5, 5, 4, 4, 7, 10, 9]

    soldierFromBaseA = soldiersA
    soldierFromBaseB = soldiersB
    soldierFromWarFieldA = soldierFromBaseA
    soldierFromWarFieldB = soldierFromBaseB

    isBoundary = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    isBoundaryNot = [0 for i in range(len(isBoundary))]
    dev = [0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    isConsistent = isBoundary

    index = [i for i in range(len(soldiersA))]
    initialSoldierA = [0 for i in range(len(soldiersA))]
    initialSoldierB = [0 for i in range(len(soldiersA))]

    random.shuffle(index)
    print(index)
    testRound = 3
    for i in range(testRound):
        temp = index[i]
        loc = index.index(i)
        index[i] = i
        index[loc] = temp
    print(index)

    mapListBound=[mapListBound[index[i]] for i in range(len(index))]
    mapListRed=[mapListRed[index[i]] for i in range(len(index))]
    mapListBlue=[mapListBlue[index[i]] for i in range(len(index))]
    soldiersA=[soldiersA[index[i]] for i in range(len(index))]
    soldiersB=[soldiersB[index[i]] for i in range(len(index))]
    soldierFromBaseA=[soldierFromBaseA[index[i]] for i in range(len(index))]
    soldierFromBaseB=[soldierFromBaseB[index[i]] for i in range(len(index))]
    soldierFromWarFieldA = [soldierFromWarFieldA[index[i]] for i in range(len(index))]
    soldierFromWarFieldB = [soldierFromWarFieldB[index[i]] for i in range(len(index))]
    isBoundary=[isBoundary[index[i]] for i in range(len(index))]
    dev=[dev[index[i]] for i in range(len(index))]
    isConsistent=[isConsistent[index[i]] for i in range(len(index))]

    coin = random.uniform(0,1)

    if isBound == 1:
        return mapListBound, isBoundary, isConsistent, dev, soldiersA, soldiersB, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB
    if isBound == 2:
        return mapListBlue, isBoundaryNot, isConsistent, dev, soldiersA, soldiersB, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB
    if isBound == 3:
        return mapListRed, isBoundaryNot, isConsistent, dev, soldiersA, soldiersB, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB
