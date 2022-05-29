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
from src.environment import checkAnnihilation, UnpackState, Observe, Transit, Reset, Terminal, CheckTerminal, \
    RewardFunction, TransitAutopeaceAnnihilation
import json

layerWidth = [32, 32]
saveAllmodels = True

def main():
    debug = 0
    if debug:
        mapSize = 9
        colorA = -1
        colorB = -1
        maxEpisode = 10000
        maxTimeStep = 25
        bufferSize = 1e4
        minibatchSize = 128
        learningRateActor = 0.01
        learningRateCritic = 0.01
        gamma = 0.95
        tau = 0.01
        learnInterval = 100
    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        mapSize = int(condition['mapSize'])
        colorA = int(condition['colorA'])
        colorB = int(condition['colorB'])
        maxEpisode = int(condition['maxEpisode'])
        maxTimeStep = int(condition['maxTimeStep'])
        bufferSize = int(condition['bufferSize'])
        minibatchSize = int(condition['minibatchSize'])
        learningRateActor = float(condition['learningRateActor'])
        learningRateCritic = float(condition['learningRateCritic'])
        gamma = float(condition['gamma'])
        tau = float(condition['tau'])
        learnInterval = int(condition['learnInterval'])

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
    rewardFunction = RewardFunction(unpackState, checkTerminal, transitAutopeaceAnnihilation, terminal)

    reset = Reset(mapSize, terminal) if colorA == -1 else Reset(mapSize, terminal, colorA, colorB)
    observe = lambda state: [Observe(unpackState, mapSize, agentID)(state) for agentID in range(numAgents)]
    actionDim = mapSize - 1
    obsShape = [len(observe(reset())[obsID]) for obsID in range(numAgents)]


    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    trainCriticBySASR = TrainCriticBySASR(actByPolicyTargetNoisyForNextState, learningRateCritic, gamma)
    trainCritic = TrainCritic(trainCriticBySASR)
    trainActorFromSA = TrainActorFromSA(learningRateActor)
    trainActor = TrainActor(trainActorFromSA)

    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)
    sampleBatchFromMemory = SampleFromMemory(minibatchSize)

    learningStartBufferSize = minibatchSize * maxTimeStep
    startLearn = StartLearn(learningStartBufferSize, learnInterval)

    trainMADDPGModels = TrainMADDPGModelsWithBuffer(updateParameters, trainActor, trainCritic, sampleBatchFromMemory, startLearn, modelsList)

    actOneStepOneModel = ActOneStepOneHot(actByPolicyTrainNoisy)
    actOneStep = lambda allAgentsStates, runTime: [actOneStepOneModel(model, allAgentsStates) for model in modelsList]
    sampleOneStep = SampleOneStep(transit, rewardFunction)
    runTimeStep = RunTimeStep(actOneStep, sampleOneStep, trainMADDPGModels, observe = observe)
    isTerminal = lambda state: terminal.terminal
    runEpisode = RunEpisode(reset, runTimeStep, maxTimeStep, isTerminal)

    getAgentModel = lambda agentId: lambda: trainMADDPGModels.getTrainedModels()[agentId]
    getModelList = [getAgentModel(i) for i in range(numAgents)]
    modelSaveRate = 5000

    fileName = "war{}gridsRandomColor{}eps{}step{}buffer{}batch{}acLR{}crLR{}gamma{}tau{}intv{}layer_agent".format(mapSize, maxEpisode, maxTimeStep, bufferSize, minibatchSize, learningRateActor, learningRateCritic, gamma, tau, learnInterval, layerWidth[0]) \
        if colorA == -1 else \
               "war{}grids{}colorA{}colorB{}eps{}step{}buffer{}batch{}acLR{}crLR{}gamma{}tau{}intv{}layer_agent".format(mapSize, colorA, colorB, maxEpisode, maxTimeStep, bufferSize, minibatchSize, learningRateActor, learningRateCritic, gamma, tau, learnInterval, layerWidth[0])
    print(fileName)
    modelDir = os.path.join(dirName, '..', 'trainedModels')
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)

    modelPath = os.path.join(modelDir, fileName)
    saveModels = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath + str(i), saveAllmodels) for i, getTrainedModel in enumerate(getModelList)]
    maddpg = RunAlgorithm(runEpisode, maxEpisode, saveModels, numAgents, printEpsFrequency=100)
    replayBuffer = getBuffer(bufferSize)
    meanRewardList = maddpg(replayBuffer)


if __name__ == '__main__':
    main()


