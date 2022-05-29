import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from src.maddpg.trainer.MADDPG import BuildMADDPGModels, ActOneStepOneHot, actByPolicyTrainNoisy
from src.loadSaveModel import saveVariables, restoreVariables, saveToPickle
from src.trajectory import SampleTrajectoryComparePolicies
from src.environment import *
from src.functionWarGamePure import CheckAutoPeace

layerWidth = [32, 32]
learningRateActor = 0.01
learningRateCritic = 0.01
gamma = 0.95
tau = 0.01
bufferSize = 1e4
minibatchSize = 128


def calcAgentsReward(traj):
    rewardIDinTraj = 2
    agentsTrajReward = np.sum([timeStepInfo[rewardIDinTraj] for timeStepInfo in traj], axis=0)
    return agentsTrajReward

def main():
    # mapSizeLevels = [8]
    # colorLevels = [(4, 4), (5, 3), (3, 5)]
    # maxEpisodeLevels = [30000]
    # maxTimeStepLevels = [25]
    # bufferSizeLevels = [10000, 100000]
    # minibatchSizeLevels = [64, 128, 256]
    # learningRateActorLevels = [0.01]
    # learningRateCriticLevels = [0.01]
    # gammaLevels = [0.95]
    # tauLevels = [0.01]
    # learnIntervalLevels = [20, 50, 100]

    mapSizeLevels = [8]
    colorLevels = [(4, 4)]
    maxEpisodeLevels = [30000]
    maxTimeStepLevels = [25]
    bufferSizeLevels = [10000]
    minibatchSizeLevels = [64]
    learningRateActorLevels = [0.01]
    learningRateCriticLevels = [0.01]
    gammaLevels = [0.95]
    tauLevels = [0.01]
    learnIntervalLevels = [100]

    conditionLevels = [(mapSize, colorA, colorB, maxEpisode, maxTimeStep, bufferSize, minibatchSize, learningRateActor,
                        learningRateCritic, gamma, tau, learnInterval)
                       for mapSize in mapSizeLevels
                       for colorA, colorB in colorLevels
                       for maxEpisode in maxEpisodeLevels
                       for maxTimeStep in maxTimeStepLevels
                       for bufferSize in bufferSizeLevels
                       for minibatchSize in minibatchSizeLevels
                       for learningRateActor in learningRateActorLevels
                       for learningRateCritic in learningRateCriticLevels
                       for gamma in gammaLevels
                       for tau in tauLevels
                       for learnInterval in learnIntervalLevels]

    for condition in conditionLevels:
        mapSize, colorA, colorB, maxEpisode, maxTimeStep, bufferSize, minibatchSize, learningRateActor, learningRateCritic, gamma, tau, learnInterval = condition

        compareWithRandom = False
        modelIDToTest = 0
        trainEps = maxEpisode #TODO: the model eps to test

        compulsoryEndTurn = maxTimeStep
        peaceEndTurn = 3
        numAgents = 2

        checkAutoPeace = CheckAutoPeace(peaceEndTurn)
        unpackState = UnpackState(mapSize)
        transit = Transit(unpackState)
        transitAutopeaceAnnihilation = TransitAutopeaceAnnihilation(compulsoryEndTurn, unpackState, transit, mapSize)

        terminal = Terminal()
        checkTerminal = CheckTerminal(compulsoryEndTurn, unpackState, checkAutoPeace, checkAnnihilation)
        rewardFunction = RewardFunction(unpackState, checkTerminal, transitAutopeaceAnnihilation, terminal)

        reset = Reset(mapSize, terminal, colorA, colorB)#, soldierFromWarFieldA, soldierFromWarFieldB)
        observe = lambda state: [Observe(unpackState, mapSize, agentID)(state) for agentID in range(numAgents)]
        actionDim = mapSize - 1
        obsShape = [len(observe(reset())[obsID]) for obsID in range(numAgents)]

        isTerminal = lambda state: terminal.terminalCheck()
        sampleTrajectory = SampleTrajectoryComparePolicies(maxTimeStep, transit, isTerminal, rewardFunction, reset)

# policy
        buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
        modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]
        dirName = os.path.dirname(__file__)
        fileName = "war{}grids{}colorA{}colorB{}eps{}step{}buffer{}batch{}acLR{}crLR{}gamma{}tau{}intv_agent".format(mapSize, colorA, colorB,
                   maxEpisode, maxTimeStep, bufferSize, minibatchSize, learningRateActor, learningRateCritic, gamma, tau, learnInterval)

        modelPaths = [os.path.join(dirName, '..', 'trainedModels', fileName + str(i)+ str(trainEps) + 'eps') for i in range(numAgents)]
        [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]
        actOneStepOneModel = ActOneStepOneHot(actByPolicyTrainNoisy)
        policyModels = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

        modelPolicy = lambda allAgentsStates: actOneStepOneModel(modelsList[modelIDToTest], allAgentsStates)
        randomPolicy = RandomPolicy(mapSize)
        policyMix = lambda allAgentsStates: [modelPolicy(observe(allAgentsStates)), randomPolicy(observe(allAgentsStates))]

        rewardListModel, rewardListMix, trajListModel, trajListMix = [],[], [],[]
        numTrajToSample = 100
        for i in range(numTrajToSample):
            trajModel, trajMix = sampleTrajectory(policyModels, policyMix)
            rewardListModel.append(calcAgentsReward(trajModel))
            rewardListMix.append(calcAgentsReward(trajMix))
            trajListModel.append(list(trajModel))
            trajListMix.append(list(trajMix))

        meanRewardModel = np.mean(rewardListModel, axis=0)
        meanRewardMix = np.mean(rewardListMix, axis=0)
        print('meanRewardModel', meanRewardModel, 'meanRewardMix ', meanRewardMix)

        # meanTrajReward = np.mean(rewardList)
        # seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
        # print('meanTrajReward', meanTrajReward, 'se ', seTrajReward)



if __name__ == '__main__':
    main()