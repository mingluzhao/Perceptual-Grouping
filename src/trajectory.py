# class SampleTrajectory:
#     def __init__(self, maxRunningSteps, transit, isTerminal, rewardFunc, reset):
#         self.maxRunningSteps = maxRunningSteps
#         self.transit = transit
#         self.isTerminal = isTerminal
#         self.rewardFunc = rewardFunc
#         self.reset = reset
#
#     def __call__(self, policy):
#         state = self.reset()
#         trajectory = []
#
#         for runningStep in range(self.maxRunningSteps):
#             action = policy(state)
#             nextState = self.transit(state, action)
#             reward = self.rewardFunc(state, action, nextState)
#             trajectory.append((state, action, reward, nextState))
#             state = nextState
#             if self.isTerminal(state):
#                 state = self.reset()
#
#         return trajectory



class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, rewardFunc, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.rewardFunc = rewardFunc
        self.reset = reset
        self.mapSize = 8 #TODO

    def __call__(self, policy):
        state = self.reset()
        trajectory = []

        for runningStep in range(self.maxRunningSteps):
            action = policy(state)
            nextState = self.transit(state, action)
            reward = self.rewardFunc(state, action, nextState)
            trajectory.append((state, action, reward, nextState))

            remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, \
            soldierFromBaseB, turn, colorA, colorB = self.unpackState(state)

            state = nextState
            print("{}: state {}, {}, {}, policy {} vs {}, reward {}".format(runningStep, remainingSoldiersA, remainingSoldiersB,
                                                                        warField, action[0].argmax(), action[1].argmax(), reward))
            if self.isTerminal(state):
                state = self.reset()
                print("terminal")
        return trajectory

    def unpackState(self, state):
        state = state[0]
        remainingSoldiersA = state[:self.mapSize]
        remainingSoldiersB = state[self.mapSize: self.mapSize*2]
        warField = state[self.mapSize*2: self.mapSize*3]
        soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn, colorA, colorB = state[self.mapSize*3:]
        return remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB, turn, colorA, colorB

