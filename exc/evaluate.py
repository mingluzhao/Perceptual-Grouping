import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from src.maddpg.trainer.MADDPG import BuildMADDPGModels, TrainCritic, TrainActor, TrainCriticBySASR, \
    TrainActorFromSA, TrainMADDPGModelsWithBuffer, ActOneStepOneHot, actByPolicyTrainNoisy, actByPolicyTargetNoisyForNextState
from src.maddpg.rlTools.RLrun import UpdateParameters, SampleOneStep, SampleFromMemory,\
    RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel, StartLearn
from src.loadSaveModel import saveVariables, restoreVariables, saveToPickle
from src.trajectory import SampleTrajectory
from src.environment import *
from src.functionWarGamePure import CheckAutoPeace

layerWidth = [32, 32]
learningRateActor = 0.01
learningRateCritic = 0.01
gamma = 0.95
tau = 0.01
bufferSize = 1e4
minibatchSize = 128


def calcRewardByTraj(traj):
    rewardIDinTraj = 2
    getWolfReward = lambda allAgentsReward: np.sum([allAgentsReward[wolfID] for wolfID in wolvesID])
    rewardList = [getWolfReward(timeStepInfo[rewardIDinTraj]) for timeStepInfo in traj]
    # print(rewardList)
    trajReward = np.sum(rewardList)
    return trajReward

def main():
    saveAllmodels = True

    maxEpisode = 1000
    maxTimeStep = 25
    mapSize = 8
    compulsoryEndTurn = 25
    peaceEndTurn = 3
    numAgents = 2
    maxRunningStepsToSample = maxTimeStep

    checkAutoPeace = CheckAutoPeace(peaceEndTurn)
    unpackState = UnpackState(mapSize)
    transit = Transit(unpackState)
    transitAutopeaceAnnihilation = TransitAutopeaceAnnihilation(compulsoryEndTurn, unpackState, transit, mapSize)

    terminal = Terminal()
    checkTerminal = CheckTerminal(compulsoryEndTurn, unpackState, checkAutoPeace, checkAnnihilation)
    rewardFunction = RewardFunction(unpackState, checkTerminal, transitAutopeaceAnnihilation, terminal)

    reset = Reset(mapSize, terminal, colorA)
    obsShape = [len(reset()[0])] * 2
    actionDim = mapSize - 1

    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]


    dirName = os.path.dirname(__file__)
    fileName = "war{}grids{}eps{}step{}buffer{}batch{}acLR{}crLR{}gamma{}tau_agent".format(mapSize, maxEpisode, maxTimeStep,
                                                                                           bufferSize, minibatchSize, learningRateActor,
                                                                                           learningRateCritic, gamma, tau)

    modelPaths = [os.path.join(dirName, '..', 'trainedModels', fileName + str(i) ) for i in range(numAgents)]
    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStepOneHot(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, allAgentsStates) for model in modelsList]

    rewardList = []
    numTrajToSample = 100
    trajList = []
    for i in range(numTrajToSample):
        traj = sampleTrajectory(policy)
        rew = calcWolvesTrajReward(traj, wolvesID)
        rewardList.append(rew)
        trajList.append(list(traj))

    meanTrajReward = np.mean(rewardList)
    seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
    print('meanTrajReward', meanTrajReward, 'se ', seTrajReward)



if __name__ == '__main__':
    main()