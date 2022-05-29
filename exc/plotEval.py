import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import pandas as pd
import matplotlib.pyplot as plt
import random

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