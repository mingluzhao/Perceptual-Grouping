import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

from subprocess import Popen, PIPE
import json
import math
import numpy as np
import itertools as it
import json

class ExcuteCodeOnConditionsParallel:
    def __init__(self, codeFileName, numSample, numCmdList):
        self.codeFileName = codeFileName
        self.numSample = numSample
        self.numCmdList = numCmdList

    def __call__(self, conditions):
        assert self.numCmdList >= len(conditions), "condition number > cmd number, use more cores or less conditions"
        numCmdListPerCondition = math.floor(self.numCmdList / len(conditions))
        if self.numSample:
            startSampleIndexes = np.arange(0, self.numSample, math.ceil(self.numSample / numCmdListPerCondition))
            endSampleIndexes = np.concatenate([startSampleIndexes[1:], [self.numSample]])
            startEndIndexesPair = zip(startSampleIndexes, endSampleIndexes)
            conditionStartEndIndexesPair = list(it.product(conditions, startEndIndexesPair))
            cmdList = [['python3', self.codeFileName, json.dumps(condition), str(startEndSampleIndex[0]), str(startEndSampleIndex[1])]
                       for condition, startEndSampleIndex in conditionStartEndIndexesPair]
        else:
            cmdList = [['python3', self.codeFileName, json.dumps(condition)]
                       for condition in conditions]
        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        for proc in processList:
            proc.communicate()
            # proc.wait()
        return cmdList

def main():
    file = open('conditionsMADDPG.json', 'r')
    allConditions = json.loads(file.read())

    currentHost = os.uname()[1]
    print(currentHost)
    condition = allConditions[currentHost]
    print(condition)

    mapSizeLevels = condition['mapSizeLevels']
    colorLevels = condition['colorLevels']
    maxEpisodeLevels = condition['maxEpisodeLevels']
    maxTimeStepLevels = condition['maxTimeStepLevels']
    bufferSizeLevels = condition['bufferSizeLevels']
    minibatchSizeLevels = condition['minibatchSizeLevels']
    learningRateActorLevels = condition['learningRateActorLevels']
    learningRateCriticLevels = condition['learningRateCriticLevels']
    gammaLevels = condition['gammaLevels']
    tauLevels = condition['tauLevels']
    learnIntervalLevels = condition['learnIntervalLevels']

    startTime = time.time()
    fileName = 'trainMADDPG.py'
    numSample = None
    numCpuToUse = int(0.8 * os.cpu_count())
    excuteCodeParallel = ExcuteCodeOnConditionsParallel(fileName, numSample, numCpuToUse)
    print("start")

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

    conditions = []
    for condition in conditionLevels:
        mapSize, colorA, colorB, maxEpisode, maxTimeStep, bufferSize, minibatchSize, learningRateActor, learningRateCritic, gamma, tau, learnInterval = condition
        parameters = {'mapSize': mapSize, 'colorA': colorA, 'colorB': colorB, 'maxEpisode': maxEpisode, 'maxTimeStep': maxTimeStep,
                      'bufferSize': bufferSize, 'minibatchSize': minibatchSize, 'learningRateActor': learningRateActor,
                      'learningRateCritic': learningRateCritic, 'gamma': gamma, 'tau': tau, 'learnInterval': learnInterval}
        conditions.append(parameters)

    cmdList = excuteCodeParallel(conditions)
    print(cmdList)

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))
    file.close()

if __name__ == '__main__':
    main()
