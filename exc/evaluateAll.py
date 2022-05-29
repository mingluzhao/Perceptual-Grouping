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
from src.functionWarGamePure import CheckAutoPeace
from src.evalFunctions import calcAgentsActionsMean, calcAgentsReward
import pandas as pd

numTrajToSample = 10
maxEpisode = 30000
maxTimeStep = 25
saveInterval = 5000


def evaluatePolicyPairs(df):
    bufferSize = df.index.get_level_values('bufferSize')[0]
    minibatchSize = df.index.get_level_values('minibatchSize')[0]
    learningRateActor = df.index.get_level_values('learningRateActor')[0]
    learningRateCritic = df.index.get_level_values('learningRateCritic')[0]
    gamma = df.index.get_level_values('gamma')[0]
    tau = df.index.get_level_values('tau')[0]
    learnInterval = df.index.get_level_values('learnInterval')[0]
    layerWidthSingle = df.index.get_level_values('layerWidth')[0].item()
    layerWidth = [layerWidthSingle, layerWidthSingle]
    trainEps = df.index.get_level_values('trainEps')[0]
    modelType1, modelType2 = df.index.get_level_values('evalSequence')[0]

    mapSize = df.index.get_level_values('mapSize')[0].item()
    colorA, colorB = df.index.get_level_values('color')[0]
    soldierFromWarFieldA, soldierFromWarFieldB = df.index.get_level_values('soldiers')[0]

    compulsoryEndTurn = maxTimeStep
    peaceEndTurn = 3
    numAgents = 2

    terminal = Terminal()
    checkAutoPeace = CheckAutoPeace(peaceEndTurn)
    unpackState = UnpackState(mapSize)
    transit = Transit(unpackState, terminal)
    transitAutopeaceAnnihilation = TransitAutopeaceAnnihilation(compulsoryEndTurn, unpackState, transit, mapSize)

    checkTerminal = CheckTerminal(compulsoryEndTurn, unpackState, checkAutoPeace, checkAnnihilation)
    rewardFunction = RewardFunction(unpackState, checkTerminal, transitAutopeaceAnnihilation, terminal)

    reset = Reset(mapSize, terminal, colorA, colorB, soldierFromWarFieldA, soldierFromWarFieldB)
    observe = lambda state: [Observe(unpackState, mapSize, agentID)(state) for agentID in range(numAgents)]
    actionDim = mapSize - 1
    obsShape = [len(observe(reset())[obsID]) for obsID in range(numAgents)]

    isTerminal = lambda state: terminal.terminal  # TODO
    sampleTrajectory = SampleTrajectory(maxTimeStep, transit, isTerminal, rewardFunction, reset)

    dirName = os.path.dirname(__file__)
    if isinstance(modelType1, int) or isinstance(modelType2, int):
        buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
        modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]
        fileName = "war{}grids{}colorA{}colorB{}eps{}step{}buffer{}batch{}acLR{}crLR{}gamma{}tau{}intv_agent".format(
            mapSize, colorA, colorB,
            maxEpisode, maxTimeStep, bufferSize, minibatchSize, learningRateActor, learningRateCritic, gamma, tau,
            learnInterval)

        modelPaths = [os.path.join(dirName, '..', 'trainedModels', fileName + str(i) + str(trainEps) + 'eps') for i in
                      range(numAgents)]
        [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]
        actOneStepOneModel = ActOneStepOneHot(actByPolicyTrainNoisy)

    randomPolicy = RandomPolicy(mapSize)
    model1Policy = lambda allAgentsStates: actOneStepOneModel(modelsList[modelType1], allAgentsStates) if isinstance(modelType1, int) else randomPolicy(allAgentsStates)
    model2Policy = lambda allAgentsStates: actOneStepOneModel(modelsList[modelType2], allAgentsStates) if isinstance(modelType2, int) else randomPolicy(allAgentsStates)
    policy = lambda allAgentsStates: [model1Policy(observe(allAgentsStates)), model2Policy(observe(allAgentsStates))]

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

    meanRewardAgent1, meanRewardAgent2  = np.mean(rewardList, axis=0)
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

    trajName = "war{}grids{}colorA{}colorB{}soldierA{}soldierB{}modelvs{}model{}eps{}step{}buffer{}batch{}acLR{}crLR{}layer{}gamma{}tau{}intv.pickle".format(
        mapSize, colorA, colorB,soldierFromWarFieldA, soldierFromWarFieldB, modelType1, modelType2,
        trainEps, maxTimeStep, bufferSize, minibatchSize, learningRateActor, learningRateCritic,layerWidthSingle, gamma, tau,
        learnInterval)

    trajPath = os.path.join(dirName, '..', 'trajectories')
    if not os.path.exists(trajPath):
        os.makedirs(trajPath)
    saveToPickle(trajList, os.path.join(trajPath, trajName))


    return pd.Series({'meanRewardAgent1': meanRewardAgent1, 'seRewardAgent1': seRewardAgent1,
                      'meanRewardAgent2': meanRewardAgent2, 'seRewardAgent2': seRewardAgent2,
                      'meanRewardTotal': meanRewardTotal, 'seRewardTotal': seRewardTotal,
                      'annihilationPercent': annihilationPercent, 'autopeacePercent': autopeacePercent,
                      'meanWar': meanWar, 'seWar': seWar,
                      'meanActionAgent1': meanActionAgent1, 'seActionAgent1': seActionAgent1,
                      'meanActionAgent2': meanActionAgent2, 'seActionAgent2': seActionAgent2
                      })

def main():
    independentVariables = dict()
    # environment parameters
    independentVariables['mapSize'] = [8]
    independentVariables['color'] = [(4, 4), (5, 3), (3, 5)]
    independentVariables['soldiers'] = [(5, 5), (10, 10), (5, 10), (10, 5)]
    # training parameters
    independentVariables['trainEps'] = np.arange(saveInterval, maxEpisode, saveInterval)
    independentVariables['evalSequence'] = [(0, 1), (0, 'random'), (1, 'random'), ('random', 0), ('random', 1), ('random', 'random')]
    independentVariables['bufferSize'] = [10000]
    independentVariables['minibatchSize'] = [64]
    independentVariables['learnInterval'] = [100]
    independentVariables['layerWidth'] = [32]
    independentVariables['learningRateActor'] = [0.01]
    independentVariables['learningRateCritic'] = [0.01]
    independentVariables['gamma'] = [0.95]
    independentVariables['tau'] = [0.01]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    resultDF = toSplitFrame.groupby(levelNames).apply(evaluatePolicyPairs)

    resultPath = os.path.join(dirName, '..', 'evalResults')
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    saveToPickle(resultDF, os.path.join(resultPath, 'evalResult.pkl'))
    resultDF.to_csv(os.path.join(resultPath, 'evalResult.csv'))

    print("Saved to ",  os.path.join(resultPath, 'evalResult.pkl'))


if __name__ == '__main__':
    main()