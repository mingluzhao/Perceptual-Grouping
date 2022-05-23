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
from src.loadSaveModel import saveVariables
from src.environment import *
from src.functionWarGamePure import CheckAutoPeace

layerWidth = [32, 32]
learningRateActor = 0.01
learningRateCritic = 0.01
gamma = 0.95
tau = 0.01
bufferSize = 1e4
minibatchSize = 128

def main():
    saveAllmodels = True

    maxEpisode = 1000
    maxTimeStep = 25
    mapSize = 8
    compulsoryEndTurn = 25
    peaceEndTurn = 3
    numAgents = 2

    checkAutoPeace = CheckAutoPeace(peaceEndTurn)
    unpackState = UnpackState(mapSize)
    transit = Transit(unpackState)
    transitAutopeaceAnnihilation = TransitAutopeaceAnnihilation(compulsoryEndTurn, unpackState, transit, mapSize)

    terminal = Terminal()
    checkTerminal = CheckTerminal(compulsoryEndTurn, unpackState, checkAutoPeace, checkAnnihilation)
    rewardFunction = RewardFunction(unpackState, checkTerminal, transitAutopeaceAnnihilation, terminal)

    reset = Reset(mapSize)
    obsShape = [len(reset()[0])] * 2
    actionDim = mapSize

    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    trainCriticBySASR = TrainCriticBySASR(actByPolicyTargetNoisyForNextState, learningRateCritic, gamma)
    trainCritic = TrainCritic(trainCriticBySASR)
    trainActorFromSA = TrainActorFromSA(learningRateActor)
    trainActor = TrainActor(trainActorFromSA)

    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)
    sampleBatchFromMemory = SampleFromMemory(minibatchSize)

    learnInterval = 100
    learningStartBufferSize = minibatchSize * maxTimeStep
    startLearn = StartLearn(learningStartBufferSize, learnInterval)

    trainMADDPGModels = TrainMADDPGModelsWithBuffer(updateParameters, trainActor, trainCritic, sampleBatchFromMemory, startLearn, modelsList)

    actOneStepOneModel = ActOneStepOneHot(actByPolicyTrainNoisy)
    actOneStep = lambda allAgentsStates, runTime: [actOneStepOneModel(model, allAgentsStates) for model in modelsList]

    sampleOneStep = SampleOneStep(transit, rewardFunction)
    runTimeStep = RunTimeStep(actOneStep, sampleOneStep, trainMADDPGModels)
    isTerminal = lambda state: terminal.terminal
    runEpisode = RunEpisode(reset, runTimeStep, maxTimeStep, isTerminal)

    getAgentModel = lambda agentId: lambda: trainMADDPGModels.getTrainedModels()[agentId]
    getModelList = [getAgentModel(i) for i in range(numAgents)]
    modelSaveRate = 100
    fileName = "war{}grids{}eps{}step{}buffer{}batch{}acLR{}crLR{}gamma{}tau_agent".format(mapSize, maxEpisode, maxTimeStep,
                                                                                           bufferSize, minibatchSize, learningRateActor,
                                                                                           learningRateCritic, gamma, tau)

    modelDir = os.path.join(dirName, '..', 'trainedModels')
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)

    modelPath = os.path.join(modelDir, fileName)
    saveModels = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath + str(i), saveAllmodels) for i, getTrainedModel in enumerate(getModelList)]
    maddpg = RunAlgorithm(runEpisode, maxEpisode, saveModels, numAgents)
    replayBuffer = getBuffer(bufferSize)
    meanRewardList = maddpg(replayBuffer)



if __name__ == '__main__':
    main()
# simple example


