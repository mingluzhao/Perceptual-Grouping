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
from src.trajectory import SampleTrajectory
from src.environment import *
from src.evalFunctions import calcAgentsActionsMean, calcAgentsReward

layerWidth = [32, 32]
numTrajToSample = 3

def main():
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
        trainEps = maxEpisode

        compulsoryEndTurn = maxTimeStep
        peaceEndTurn = 3
        numAgents = 2

        if mapSize % 2 == 0:
            from src.functionWarGamePure import CheckAutoPeace, calculateRemainingSoldiers
        else:
            from src.functionWarGameSevenGrid import CheckAutoPeace, calculateRemainingSoldiers

        terminal = Terminal()
        checkAutoPeace = CheckAutoPeace(peaceEndTurn)
        unpackState = UnpackState(mapSize)
        transit = Transit(unpackState, terminal, calculateRemainingSoldiers)
        transitAutopeaceAnnihilation = TransitAutopeaceAnnihilation(compulsoryEndTurn, unpackState, transit, mapSize)

        checkTerminal = CheckTerminal(compulsoryEndTurn, unpackState, checkAutoPeace, checkAnnihilation)
        getChangeInSoldiers = GetChangeInSoldiers(unpackState)
        rewardFunction = RewardFunction(checkTerminal, transitAutopeaceAnnihilation, terminal, getChangeInSoldiers)

        reset = Reset(mapSize, terminal, colorA, colorB)
        observe = lambda state: [Observe(unpackState, mapSize, agentID)(state) for agentID in range(numAgents)]
        actionDim = mapSize - 1
        obsShape = [len(observe(reset())[obsID]) for obsID in range(numAgents)]

        isTerminal = lambda state: terminal.terminal #TODO
        sampleTrajectory = SampleTrajectory(maxTimeStep, transit, isTerminal, rewardFunction, reset)

        buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
        modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]
        dirName = os.path.dirname(__file__)
        fileName = "war{}grids{}colorA{}colorB{}eps{}step{}buffer{}batch{}acLR{}crLR{}gamma{}tau{}intv_agent".format(mapSize, colorA, colorB,
                   maxEpisode, maxTimeStep, bufferSize, minibatchSize, learningRateActor, learningRateCritic, gamma, tau, learnInterval)

        modelPaths = [os.path.join(dirName, '..', 'trainedModels', fileName + str(i)+ str(trainEps) + 'eps') for i in range(numAgents)]
        [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]
        actOneStepOneModel = ActOneStepOneHot(actByPolicyTrainNoisy)
        policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

        rewardList, trajList = [], []
        annihilationList, autopeaceList, warList = [], [], []
        agent1Actions, agent2Actions = [], []

        for i in range(numTrajToSample):
            traj = sampleTrajectory(policy)
            annihilationList.append(terminal.annihilationCount)
            autopeaceList.append(terminal.autoPeaceCount)
            warList.append(terminal.warCount)

            rew = calcAgentsReward(traj)
            agent1ActionMean, agent2ActionMean = calcAgentsActionsMean(traj)

            rewardList.append(rew)
            agent1Actions.append(agent1ActionMean)
            agent2Actions.append(agent2ActionMean)
            trajList.append(list(traj))

        meanRewardAgent1, meanRewardAgent2 = np.mean(rewardList, axis=0)
        totalrewardList = np.sum(rewardList, axis=1)
        meanRewardTotal = np.mean(totalrewardList)
        seRewardAgent1, seRewardAgent2 = np.std(rewardList, axis=0) / np.sqrt(len(rewardList) - 1)
        seRewardTotal = np.std(totalrewardList) / np.sqrt(len(totalrewardList) - 1)

        annihilationPercent = np.mean(annihilationList)
        autopeacePercent = np.mean(autopeaceList)
        meanWar = np.mean(warList)
        seWar = np.std(warList) / np.sqrt(len(warList) - 1)

        meanActionAgent1 = np.mean(agent1Actions)
        seActionAgent1 = np.std(agent1Actions) / np.sqrt(len(agent1Actions) - 1)
        meanActionAgent2 = np.mean(agent2Actions)
        seActionAgent2 = np.std(agent2Actions) / np.sqrt(len(agent2Actions) - 1)

        print(dict({'meanRewardAgent1': meanRewardAgent1, 'seRewardAgent1': seRewardAgent1,
                    'meanRewardAgent2': meanRewardAgent2, 'seRewardAgent2': seRewardAgent2,
                    'meanRewardTotal': meanRewardTotal, 'seRewardTotal': seRewardTotal,
                    'annihilationPercent': annihilationPercent, 'autopeacePercent': autopeacePercent,
                    'meanWar': meanWar, 'seWar': seWar,
                    'meanActionAgent1': meanActionAgent1, 'seActionAgent1': seActionAgent1,
                    'meanActionAgent2': meanActionAgent2, 'seActionAgent2': seActionAgent2
                    }))



if __name__ == '__main__':
    main()