import os
import sys
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))
import random
import pandas as pd
import random
import math
from pygame.locals import *
from sys import exit
from src.functionWarGamePure import *


class SimulateCPEPOneStep:

    def __init__(self, soldiersFromBaseA, soldiersFromBaseB, soldiersFromField, threshold, mapSize, isEven):
        self.soldiersFromBaseA = soldiersFromBaseA
        self.soldiersFromBaseB = soldiersFromBaseB
        self.soldiersFromField = soldiersFromField
        self.possibleCPEP = [0 for _ in range(mapSize * 2 - isEven - 1)]
        self.threshold = threshold

    def __call__(self, warField, remainingSoldiersA, remainingSoldiersB, soldiersA, soldiersB):
        length = len(warField)
        for assumedPeacePoint in range(length):
            flag = 0
            for i in range(assumedPeacePoint):
                if warField[i] == 2:
                    flag = 1
            for i in range(length-assumedPeacePoint-1):
                if warField[assumedPeacePoint+i+1] == 1:
                    flag = 1
            if flag == 1:
                continue
            steadySoldierA, steadySoldierB = calculateSteadySoldier(warField, remainingSoldiersA, remainingSoldiersB)
            if sum(steadySoldierA) > soldiersA or (sum(steadySoldierA) > warField.count(1) * self.soldiersFromField + self.soldiersFromBaseA) \
                or sum(steadySoldierB) > soldiersB or (sum(steadySoldierB) > warField.count(2)*self.soldiersFromField+self.soldiersFromBaseB):
                continue
            soldierInPeacePoint = warField.count(1) * self.soldiersFromField + self.soldiersFromBaseA + remainingSoldiersA[assumedPeacePoint] - sum(steadySoldierA) \
            - (warField.count(2) * self.soldiersFromField + self.soldiersFromBaseB + remainingSoldiersB[assumedPeacePoint] - sum(steadySoldierB))
            if soldierInPeacePoint >= 0:
                if abs(soldierInPeacePoint*(length - assumedPeacePoint)/(length + 1) - remainingSoldiersA[assumedPeacePoint]) <= self.threshold:
                    self.possibleCPEP[assumedPeacePoint] += 1
            if soldierInPeacePoint < 0:
                if abs(soldierInPeacePoint*(assumedPeacePoint + 1)/(length + 1)*(-1) - remainingSoldiersB[assumedPeacePoint]) <= self.threshold:
                    self.possibleCPEP[assumedPeacePoint] += 1
        return self.possibleCPEP


def calculateSteadySoldier(warField, remainingSoldiersA, remainingSoldiersB):
    steadySoldierA = []
    steadySoldierB = []
    l = len(warField)
    for i in range(l):
        steadySoldierA.append(math.ceil((1+i)/(l-i)*remainingSoldiersA[i]))
        steadySoldierB.append(math.ceil((l-i)/(i+1)*remainingSoldiersB[i]))

    return steadySoldierA, steadySoldierB


def randomPolicy(soldiersA, soldiersB, length, shuffleFreq):
    policyA = []
    policyB = []
    for i in range(length):
        if soldiersA > 0:
            tempA = np.random.randint(0, int(soldiersA/2))
            policyA.append(tempA)
            soldiersA -= tempA
        else:
            policyA.append(0)
        if soldiersB > 0:
            tempB = np.random.randint(0, int(soldiersB/2))
            policyB.append(tempB)
            soldiersB -= tempB
        else:
            policyB.append(0)
    policyB.reverse()
    if np.random.randint(0, shuffleFreq) == 1:
        random.shuffle(policyA)
    if np.random.randint(0, shuffleFreq) == 1:
        random.shuffle(policyB)

    return policyA, policyB


def normalize(possibleCPEP):
    Z = sum(possibleCPEP)
    result = [i / Z for i in possibleCPEP]
    return result


def calculateExpectation(possibleCPEP):
    result = 0
    for i in range(len(possibleCPEP)):
        result += possibleCPEP[i]*(i-(len(possibleCPEP)-1)/2)
    return result


class SimulateCPEP:

    def __init__(self, mapSize, isEven, isSymmetrical, episode, step, simulateCPEPOneStep, soldiersA, soldiersB):
        self.mapSize = mapSize
        self.isEven = isEven
        self.isSymmetrical = isSymmetrical
        self.episode = episode
        self.step = step
        self.simulateCPEPOneStep = simulateCPEPOneStep
        self.baseLocationRandom = random.randint(1, mapSize)
        self.length = mapSize * 2 - isEven - 1
        self.soldiersA = soldiersA
        self.soldiersB = soldiersB

    def __call__(self, soldierFromBaseA, soldierFromBaseB, soldierFromWarField, shuffleFreq, threshold):
        for e in range(self.episode):
            remainingSoldiersA = [0 for i in range(self.mapSize * 2 - self.isEven - 1)]
            remainingSoldiersB = [0 for i in range(self.mapSize * 2 - self.isEven - 1)]
            soldiersA = self.soldiersA
            soldiersB = self.soldiersB
            for s in range(self.step):
                policyA, policyB = randomPolicy(soldiersA, soldiersB, self.length, shuffleFreq)
                warField = judgeResult(policyA, policyB, remainingSoldiersA, remainingSoldiersB)
                [remainingSoldiersA, remainingSoldiersB] = \
                    calculateRemainingSoldiers(policyA, policyB, remainingSoldiersA, remainingSoldiersB,
                                               self.baseLocationRandom)
                [soldiersA, soldiersB, soldiersGained] = \
                    calculateSoldiers(warField, soldierFromBaseA, soldierFromBaseB, soldiersA - sum(policyA),
                                      soldiersB - sum(policyB), self.baseLocationRandom, soldierFromWarField)
                possibleCPEP = self.simulateCPEPOneStep(warField, remainingSoldiersA, remainingSoldiersB, soldiersA,
                                                   soldiersB)
        possibleCPEP = normalize(possibleCPEP)
        return possibleCPEP


class Simulation:

    def __init__(self, episode, step, length, shuffleFreq, soldiersA, soldiersB):
        self.episode = episode
        self.step = step
        self.length = length
        self.shuffleFreq = shuffleFreq
        self.soldiersA = soldiersA
        self.soldiersB = soldiersB

    def __call__(self, soldierFromWarField, soldierFromBaseA, soldierFromBaseB, simulationOneStep, baseLocationRandom):
        for e in range(self.episode):
            policyA = [0 for i in range(self.length)]
            policyB = [0 for i in range(self.length)]
            warField = [0 for i in range(self.length)]
            remainingSoldiersA = [0 for i in range(self.length)]
            remainingSoldiersB = [0 for i in range(self.length)]
            soldiersA = self.soldiersA
            soldiersB = self.soldiersB
            for s in range(self.step):
                policyA, policyB = randomPolicy(soldiersA, soldiersB, self.length, self.shuffleFreq)
                warField = judgeResult(policyA, policyB, remainingSoldiersA, remainingSoldiersB)
                [remainingSoldiersA, remainingSoldiersB] = \
                    calculateRemainingSoldiers(policyA, policyB, remainingSoldiersA, remainingSoldiersB,
                                               baseLocationRandom)
                [soldiersA, soldiersB, soldiersGained] = \
                    calculateSoldiers(warField, soldierFromBaseA, soldierFromBaseB, soldiersA - sum(policyA),
                                      soldiersB - sum(policyB), baseLocationRandom, soldierFromWarField)
                possibleCPEP = simulationOneStep(warField, remainingSoldiersA, remainingSoldiersB, soldiersA,
                                                    soldiersB)

        return normalize(possibleCPEP)




