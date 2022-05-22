from src.functionWarGamePure import *
peaceEndTurn = 3
checkAutoPeace = CheckAutoPeace(peaceEndTurn)

calculateRemainingSoldiers(policyA, policyB, remainingSoldiersA, remainingSoldiersB, warField,
                               soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB)


# transition

def transit(policyA, policyB, remainingSoldiersA, remainingSoldiersB, warField,
                               soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB)
    remainingSoldiersA, remainingSoldiersB, warField,
    soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB = calculateRemainingSoldiers(policyA, policyB, remainingSoldiersA, remainingSoldiersB, warField,
                               soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB)

# reset():
    # return state

# isTerminal(state):
    # return True or False

# reward(state):
    # return 0 if not terminal, return soldierNumber*0.3



from src.functionWarGamePure import *



compulsoryEndTurn = 25
peaceEndTurn = 3
checkAutoPeace = CheckAutoPeace(peaceEndTurn)

def checkAnnihilation(warField):
    if warField.count(1) == len(warField) - 1 or warField.count(2) == len(warField) - 1:
        return 1
    else:
        return 0

def transition(policyA, policyB, remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn):
    remainingSoldiersA, remainingSoldiersB, warField, warLocation = calculateRemainingSoldiers(policyA, policyB, remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB)
    turn += 1
    return remainingSoldiersA, remainingSoldiersB, warField, turn

def reset(mapSize):
    # random soldierFromWarField for each episode, can be constant
    soldierFromWarFieldA = random.randint(3, 10)
    soldierFromWarFieldB = random.randint(3, 10)
    soldierFromBaseA = soldierFromWarFieldA
    soldierFromBaseB = soldierFromWarFieldB

    remainingSoldiersA = [0 for i in range(mapSize)]
    remainingSoldiersB = [0 for i in range(mapSize)]
    remainingSoldiersA[0] = soldierFromBaseA
    remainingSoldiersB[-1] = soldierFromBaseB

    warField = [0 for i in range(mapSize)]
    warField[0] = 1
    warField[-1] = 2

    turn = 0
    return remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn

def checkTerminal(policyA, policyB, warField, turn, compulsoryEndTurn):
    isAutoPeace = checkAutoPeace(policyA, policyB, warField)
    isAnnihilation = checkAnnihilation(warField)
    if turn >= compulsoryEndTurn or isAutoPeace or isAnnihilation:
        isTerminal = 1
    else:
        isTerminal = 0
    return isTerminal, isAutoPeace, isAnnihilation

def rewardFunction(remainingSoldiersA, remainingSoldiersB, isTerminal):
    if isTerminal:
        rewardA = sum(remainingSoldiersA)
        rewardB = sum(remainingSoldiersB)
        return rewardA, rewardB
    else:
        return 0, 0

def randomPolicy(mapSize):
    policy = [0 for i in range(mapSize)]
    policy[random.randint(0,mapSize-1)] = 1
    return policy

# simple example
maxEpisode = 100
mapSize = 8
rewardList = []
for i in range(maxEpisode):
    remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn = reset(mapSize)
    while(True):
        policyA = randomPolicy(mapSize)
        policyB = randomPolicy(mapSize)
        remainingSoldiersA, remainingSoldiersB, warField, turn = transition(policyA, policyB, remainingSoldiersA, remainingSoldiersB, warField,
                                                                            soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn)
        isTerminal, isAutoPeace, isAnnihilation = checkTerminal(policyA, policyB, warField, turn, compulsoryEndTurn)
        # handle special case
        if isAutoPeace or isAnnihilation:
            remainingTurn = compulsoryEndTurn-turn
            for j in range(remainingTurn):
                policyA = [0 for i in range(mapSize)]
                policyA[0] = 1
                policyB = [0 for i in range(mapSize)]
                policyB[-1] = 1
                remainingSoldiersA, remainingSoldiersB, warField, turn = transition(policyA, policyB,
                                                                                    remainingSoldiersA,
                                                                                    remainingSoldiersB, warField,
                                                                                    soldierFromWarFieldA,
                                                                                    soldierFromWarFieldB,
                                                                                    soldierFromBaseA, soldierFromBaseB,
                                                                                    turn)
        rewardA, rewardB = rewardFunction(remainingSoldiersA, remainingSoldiersB, isTerminal)
        if isTerminal:
            rewardList.append(rewardA)
            break

    print('episode: {}, rewardA: {}'.format(i, rewardList[i]))

