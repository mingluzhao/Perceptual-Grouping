from src.functionWarGamePure import calculateRemainingSoldiers
import numpy as np
import random


def checkAnnihilation(warField):
    warField = list(warField)
    if warField.count(1) == len(warField) - 1 or warField.count(2) == len(warField) - 1:
        return 1
    else:
        return 0


class UnpackState:
    def __init__(self, mapSize):
        self.mapSize = mapSize

    def __call__(self, state):
        state = state[0]
        remainingSoldiersA = state[:self.mapSize]
        remainingSoldiersB = state[self.mapSize: self.mapSize*2]
        warField = state[self.mapSize*2: self.mapSize*3]
        soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn = state[self.mapSize*3:]
        return remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn


class Transit:
    def __init__(self, unpackState):
        self.unpackState = unpackState

    def __call__(self, state, policy):
        policyA, policyB = policy
        remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn = self.unpackState(state)
        remainingSoldiersA, remainingSoldiersB, warField, warLocation = calculateRemainingSoldiers(policyA, policyB, remainingSoldiersA, remainingSoldiersB,
                                                                                                   warField, soldierFromWarFieldA, soldierFromWarFieldB,
                                                                                                   soldierFromBaseA, soldierFromBaseB)
        turn += 1
        nextState = list(remainingSoldiersA) + list(remainingSoldiersB) + list(warField) + [soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA,
                                                                                            soldierFromBaseB, turn]
        nextStateArray = [np.array(nextState), np.array(nextState)]
        return nextStateArray


class Reset:
    def __init__(self, mapSize):
        self.mapSize = mapSize

    def __call__(self):
        # random soldierFromWarField for each episode, can be constant
        soldierFromWarFieldA = random.randint(3, 10)
        soldierFromWarFieldB = random.randint(3, 10)
        soldierFromBaseA = soldierFromWarFieldA
        soldierFromBaseB = soldierFromWarFieldB

        remainingSoldiersA = [0 for i in range(self.mapSize)]
        remainingSoldiersB = [0 for i in range(self.mapSize)]
        remainingSoldiersA[0] = soldierFromBaseA
        remainingSoldiersB[-1] = soldierFromBaseB

        warField = [0 for i in range(self.mapSize)]
        warField[0] = 1
        warField[-1] = 2

        turn = 0
        state = list(remainingSoldiersA) + list(remainingSoldiersB) + list(warField) + [soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn]
        return [np.array(state), np.array(state)]


class Terminal(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.terminal = False
        self.autoPeace = False
        self.annihilation = False

    def isAutoPeace(self):
        self.terminal = True
        self.autoPeace = True

    def isAnnihilation(self):
        self.terminal = True
        self.annihilation = True

    def isTerminal(self):
        self.terminal = True


class CheckTerminal:
    def __init__(self, compulsoryEndTurn, unpackState, checkAutoPeace, checkAnnihilation):
        self.compulsoryEndTurn = compulsoryEndTurn
        self.unpackState = unpackState
        self.checkAutoPeace = checkAutoPeace
        self.checkAnnihilation = checkAnnihilation

    def __call__(self, policy, state):
        policyA, policyB = policy
        remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn = self.unpackState(state)
        isAutoPeace = self.checkAutoPeace(policyA, policyB, warField)
        isAnnihilation = self.checkAnnihilation(warField)
        isTerminal = True if turn >= self.compulsoryEndTurn or isAutoPeace or isAnnihilation else False

        return isTerminal, isAutoPeace, isAnnihilation


class RewardFunction:
    def __init__(self, unpackState, checkTerminal, transitAutopeaceAnnihilation, terminal):
        self.unpackState = unpackState
        self.checkTerminal = checkTerminal
        self.terminal = terminal
        self.transitAutopeaceAnnihilation = transitAutopeaceAnnihilation

    def __call__(self, state, policy, nextState):
        remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn = self.unpackState(state)
        self.terminal.terminal, self.terminal.autoPeace, self.terminal.annihilation = self.checkTerminal(policy, state)

        if self.terminal.autoPeace or self.terminal.annihilation: # move forward to get rewards automatically
            state = self.transitAutopeaceAnnihilation(state)
            remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn = self.unpackState(state)

        if self.terminal.isTerminal:
            rewardA = sum(remainingSoldiersA)
            rewardB = sum(remainingSoldiersB)
            return [rewardA, rewardB]
        else:
            return [0, 0]


def randomPolicy(mapSize):
    policy = [0 for i in range(mapSize)]
    policy[random.randint(0, mapSize-1)] = 1
    return policy


class TransitAutopeaceAnnihilation:
    def __init__(self, compulsoryEndTurn, unpackState, transit, mapSize):
        self.compulsoryEndTurn = compulsoryEndTurn
        self.unpackState = unpackState
        self.mapSize = mapSize
        self.transit = transit

    def __call__(self, state):
        remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn = self.unpackState(state)
        remainingTurn = self.compulsoryEndTurn - turn

        for j in range(remainingTurn):
            policyA = [0 for i in range(self.mapSize)]
            policyA[0] = 1
            policyB = [0 for i in range(self.mapSize)]
            policyB[-1] = 1
            policy = [policyA, policyB]
            state = self.transit(state, policy)

        return state