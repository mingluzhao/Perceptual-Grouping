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
        soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn, colorA, colorB = state[self.mapSize*3:]
        return remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn, colorA, colorB


class Observe:
    def __init__(self, unpackState, mapSize, agentID): # color from agent's perspective
        self.unpackState = unpackState
        self.mapSize = mapSize
        self.agentID = agentID

    def __call__(self, state):
        remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn, colorA, colorB = self.unpackState(state)
        # 0 as red, 1 as blue
        if self.agentID == 0:
            colorList = [0]* colorA + [1]*(self.mapSize - colorA)
        else:
            colorList = [0] * (self.mapSize - colorB) + [1]* colorB
        agentObs = list(state[self.agentID][:-2]) + list(colorList) # remove color number

        return agentObs


class Transit:
    def __init__(self, unpackState):
        self.unpackState = unpackState

    def __call__(self, state, policy):
        policyA, policyB = policy
        policyA = list(policyA) + [0]
        policyB = [0] + list(policyB)

        remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn, colorA, colorB = self.unpackState(state)
        remainingSoldiersA, remainingSoldiersB, warField, warLocation = calculateRemainingSoldiers(policyA, policyB, remainingSoldiersA, remainingSoldiersB,
                                                                                                   warField, soldierFromWarFieldA, soldierFromWarFieldB,
                                                                                                   soldierFromBaseA, soldierFromBaseB)
        turn += 1
        nextState = list(remainingSoldiersA) + list(remainingSoldiersB) + list(warField) + [soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn, colorA, colorB]
        nextStateArray = [np.array(nextState), np.array(nextState)]
        return nextStateArray


class Reset:
    def __init__(self, mapSize, terminal, colorA, colorB):
        self.mapSize = mapSize
        self.terminal = terminal
        self.colorA = colorA
        self.colorB = colorB

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
        # colorAGrids = [1]* self.colorA + [0] * self.colorB
        # colorBGrids = [0]* self.colorA + [1] * self.colorB

        # state = list(remainingSoldiersA) + list(remainingSoldiersB) + list(warField) + \
        #         list(colorAGrids) + list(colorBGrids) + \
        #         [soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn]

        state = list(remainingSoldiersA) + list(remainingSoldiersB) + list(warField) + \
                [soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn,
                 self.colorA, self.colorB]

        self.terminal.reset()
        return [np.array(state), np.array(state)]


class Terminal(object):
    def __init__(self):
        self.reset()

    def reset(self): # used at the start of each episode
        # print("before reset: {} autopeace, {} annihilation".format(self.autoPeaceCount, self.annihilationCount))
        # print("reset terminal")
        self.terminal = False
        self.autoPeace = False
        self.annihilation = False
        self.autoPeaceCount = 0
        self.annihilationCount = 0

    def isAutoPeace(self):
        self.terminal = True
        self.autoPeace = True
        self.autoPeaceCount += 1
        # print("Auto peace, now {} times".format(self.autoPeaceCount))

    def isAnnihilation(self):
        self.terminal = True
        self.annihilation = True
        self.annihilationCount += 1
        # print("Annihilation, now {} times".format(self.annihilationCount))

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
        policyA = list(policyA) + [0]
        policyB = [0] + list(policyB)

        remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn, colorA, colorB = self.unpackState(state)
        isAutoPeace = self.checkAutoPeace(policyA, policyB, warField)
        isAnnihilation = self.checkAnnihilation(warField)
        isTerminal = True if turn+1 >= self.compulsoryEndTurn or isAutoPeace or isAnnihilation else False

        if isAutoPeace:
            print("AutoPeace")
        if isAnnihilation:
            print("isAnnihilation")
        return isTerminal, isAutoPeace, isAnnihilation


class RewardFunction:
    def __init__(self, unpackState, checkTerminal, transitAutopeaceAnnihilation, terminal):
        self.unpackState = unpackState
        self.checkTerminal = checkTerminal
        self.terminal = terminal
        self.transitAutopeaceAnnihilation = transitAutopeaceAnnihilation

    def __call__(self, state, policy, nextState):
        terminal, autoPeace, annihilation = self.checkTerminal(policy, state)

        if autoPeace:
            self.terminal.isAutoPeace()
        if annihilation:
            self.terminal.isAnnihilation()
        if terminal:
            self.terminal.isTerminal()

        if self.terminal.terminal: # move forward to get rewards automatically
            state = self.transitAutopeaceAnnihilation(state)
            remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn, colorA, colorB = self.unpackState(state)
            rewardA = sum(remainingSoldiersA)
            rewardB = sum(remainingSoldiersB)
            reward = [rewardA, rewardB]
        else:
            reward = [0, 0]
        # print("state {}, {}, {}, policy {} vs {}, reward {}".format(remainingSoldiersA, remainingSoldiersB,
        #                                                             warField, policy[0].argmax(), policy[1].argmax(), reward))
        return reward


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
        remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn, colorA, colorB = self.unpackState(state)
        remainingTurn = self.compulsoryEndTurn - turn

        for j in range(remainingTurn):
            policyA = [0 for i in range(self.mapSize-1)]
            policyA[0] = 1
            policyB = [0 for i in range(self.mapSize-1)]
            policyB[-1] = 1
            policy = [policyA, policyB]
            state = self.transit(state, policy)

        return state