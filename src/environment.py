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
    def __init__(self, unpackState, terminal):
        self.unpackState = unpackState
        self.terminal = terminal

    def __call__(self, state, policy):
        policyA, policyB = policy
        policyA = list(policyA) + [0]
        policyB = [0] + list(policyB)

        remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn, colorA, colorB = self.unpackState(state)
        remainingSoldiersA, remainingSoldiersB, warField, warLocation = calculateRemainingSoldiers(policyA, policyB, remainingSoldiersA, remainingSoldiersB,
                                                                                                   warField, soldierFromWarFieldA, soldierFromWarFieldB,
                                                                                                   soldierFromBaseA, soldierFromBaseB)
        if warLocation>0: # isWar # TODO: for evaluation
            self.terminal.isWar(warLocation)

        turn += 1
        nextState = list(remainingSoldiersA) + list(remainingSoldiersB) + list(warField) + [soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn, colorA, colorB]
        nextStateArray = [np.array(nextState), np.array(nextState)]
        return nextStateArray

# policyA = [1] + [0]*7
# policyB = [0]*7 + [1]
# remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, \
# soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn, colorA, colorB = unpackState(state0)
# calculateRemainingSoldiers(policyA, policyB, remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB)
#



class Reset:
    def __init__(self, mapSize, terminal, colorA, colorB, soldierFromWarFieldA = -1, soldierFromWarFieldB= -1):
        self.mapSize = mapSize
        self.terminal = terminal
        self.colorA = colorA
        self.colorB = colorB
        self.soldierFromWarFieldA = soldierFromWarFieldA
        self.soldierFromWarFieldB = soldierFromWarFieldB
        self.soldiersFromWarFieldAList = [10, 9, 8, 7, 6, 5, 7, 6, 7, 10, 9, 7, 10, 9, 8, 3, 5, 4]
        self.soldiersFromWarFieldBList = [10, 9, 8, 7, 6, 5, 9, 8, 10, 8, 6, 5, 5, 4, 4, 7, 10, 9]

    def __call__(self):
        if self.soldierFromWarFieldA != -1 and self.soldierFromWarFieldB != -1:
            # specify reset state
            soldierFromWarFieldA = self.soldierFromWarFieldA
            soldierFromWarFieldB = self.soldierFromWarFieldB
        else:
            # pseudorandom assignment of sodier number - align with human experiment
            coin = random.randint(0, len(self.soldiersFromWarFieldAList)-1)
            soldierFromWarFieldA = self.soldiersFromWarFieldAList[coin]
            soldierFromWarFieldB = self.soldiersFromWarFieldBList[coin]

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
        self.warCount = 0
        self.warLocation = []

    def isWar(self, location):
        self.warCount += 1
        self.warLocation.append(location)
        print("War number {}, at {}".format(self.warCount, location))

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

    def terminalCheck(self):
        if self.terminal:
            self.reset()
            return True
        else:
            return False


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
            print("peace")
        if annihilation:
            self.terminal.isAnnihilation()
            print("annihilation")

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



class RandomPolicy:
    def __init__(self, mapSize):
        self.mapSize = mapSize

    def __call__(self, allAgentsStates):
        actionTaken = random.randint(0, self.mapSize - 2)
        policy = [0]* (self.mapSize - 1)
        policy[actionTaken] = 1
        return np.array(policy)


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

#
# transitAutopeaceAnnihilation()
#
# state0 = [np.array([43, 32, 27, 22, 17,  8,  7,  0,  0,  0,  0,  0,  0,  0,  0,  5,  1,
#         1,  1,  1,  1,  1,  1,  2,  7,  5,  7,  5, 19,  4,  4]), np.array([43, 32, 27, 22, 17,  8,  7,  0,  0,  0,  0,  0,  0,  0,  0,  5,  1,
#         1,  1,  1,  1,  1,  1,  2,  7,  5,  7,  5, 19,  4,  4])]
#
# state1 = [np.array([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 1, 0, 0, 0, 0, 0,
#        0, 2, 7, 9, 7, 9, 0, 4, 4]), np.array([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 1, 0, 0, 0, 0, 0,
#        0, 2, 7, 9, 7, 9, 0, 4, 4])]