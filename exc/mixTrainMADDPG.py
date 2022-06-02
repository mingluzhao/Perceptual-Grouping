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
    RewardFunction, TransitAutopeaceAnnihilation, GetChangeInSoldiers
import json
import numpy as np

layerWidth = [32, 32]
saveAllmodels = True
totalModels = 20

def shufflePairs(firstModelNumber, secondModelNumber):
    firstModelOrder = list(range(firstModelNumber))
    secondModelOrder = list(range(firstModelNumber, firstModelNumber + secondModelNumber))
    np.random.shuffle(firstModelOrder)
    np.random.shuffle(secondModelOrder)
    return list(zip(firstModelOrder, secondModelOrder))

def main():
    debug = 1
    if debug:
        mapSize = 9
        colorA = -1
        colorB = -1
        maxEpisode = 200000
        maxTimeStep = 25
        bufferSize = 1e4
        minibatchSize = 128
        learningRateActor = 0.01
        learningRateCritic = 0.01
        gamma = 0.95
        tau = 0.01
        learnInterval = 100
        switchInterval = 200  # play 200 episodes then switch partner
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
        switchInterval = int(condition['switchInterval'])  # play 200 episodes then switch partner


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

    reset = Reset(mapSize, terminal) if colorA == -1 else Reset(mapSize, terminal, colorA, colorB)
    observe = lambda state: [Observe(unpackState, mapSize, agentID)(state) for agentID in range(numAgents)]
    actionDim = mapSize - 1
    obsShape = [len(observe(reset())[obsID]) for obsID in range(numAgents)]
    isTerminal = lambda state: terminal.terminal

    ################################### model ###################################
    trainCriticBySASR = TrainCriticBySASR(actByPolicyTargetNoisyForNextState, learningRateCritic, gamma)
    trainCritic = TrainCritic(trainCriticBySASR)
    trainActorFromSA = TrainActorFromSA(learningRateActor)
    trainActor = TrainActor(trainActorFromSA)

    paramUpdateInterval = 1  #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)
    sampleBatchFromMemory = SampleFromMemory(minibatchSize)

    actOneStepOneModel = ActOneStepOneHot(actByPolicyTrainNoisy)
    actOneStep = lambda allAgentsStates, runTime: [actOneStepOneModel(model, allAgentsStates) for model in modelsList]
    sampleOneStep = SampleOneStep(transit, rewardFunction)

    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelPairsNum = int(totalModels/2)
    allModelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents) for _ in range(modelPairsNum)]
    getFileName =lambda modelID: "war{}gridsRandomColor{}eps{}step{}buffer{}batch{}acLR{}crLR{}gamma{}tau{}intv{}layer{}switch_model{}".format(mapSize, maxEpisode, maxTimeStep, bufferSize, minibatchSize, learningRateActor, learningRateCritic, gamma, tau, learnInterval, layerWidth[0], switchInterval, modelID) \
        if colorA == -1 else "war{}grids{}colorA{}colorB{}eps{}step{}buffer{}batch{}acLR{}crLR{}gamma{}tau{}intv{}layer{}switch_model{}".format(mapSize, colorA, colorB, maxEpisode, maxTimeStep, bufferSize, minibatchSize, learningRateActor, learningRateCritic, gamma, tau, learnInterval, layerWidth[0], switchInterval, modelID)
    firstModelNumber = modelPairsNum
    secondModelNumber = modelPairsNum

    modelDir = os.path.join(dirName, '..', 'trainedModels', 'mixTrain')
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)

    for shuffleID in range(int(maxEpisode/switchInterval)):
        print("Shuffle {} out of {} interactions".format(shuffleID, int(maxEpisode/switchInterval)))
        modelOrder = shufflePairs(firstModelNumber, secondModelNumber)

        for model1ID, model2ID in np.reshape(modelOrder, (-1, 2)):
            print("Training model {} and {}".format(model1ID, model2ID))
            modelsList = [allModelsList[model1ID], allModelsList[model2ID]]
            learningStartBufferSize = minibatchSize * maxTimeStep
            startLearn = StartLearn(learningStartBufferSize, learnInterval)
            trainMADDPGModels = TrainMADDPGModelsWithBuffer(updateParameters, trainActor, trainCritic, sampleBatchFromMemory, startLearn, modelsList)

            runTimeStep = RunTimeStep(actOneStep, sampleOneStep, trainMADDPGModels, observe=observe)
            runEpisode = RunEpisode(reset, runTimeStep, maxTimeStep, isTerminal)

            getAgentModel = lambda agentId: lambda: trainMADDPGModels.getTrainedModels()[agentId]
            modelSaveRate = switchInterval
            modelPath1 = os.path.join(modelDir, getFileName(model1ID) + "_shuffle" + str(shuffleID)+ "_")
            modelPath2 = os.path.join(modelDir, getFileName(model2ID) + "_shuffle" + str(shuffleID)+ "_")
            saveModels = [SaveModel(modelSaveRate, saveVariables, getAgentModel(0), modelPath1, saveAllmodels),
                          SaveModel(modelSaveRate, saveVariables, getAgentModel(1), modelPath2, saveAllmodels) ]
            maxEpisodeEachInteraction = switchInterval
            maddpg = RunAlgorithm(runEpisode, maxEpisodeEachInteraction, saveModels, numAgents, printEpsFrequency=1 )
            replayBuffer = getBuffer(bufferSize)
            maddpg(replayBuffer)

            allModelsList[model1ID] = getAgentModel(0)()
            allModelsList[model2ID] = getAgentModel(1)()


if __name__ == '__main__':
    main()


