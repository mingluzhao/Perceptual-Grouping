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
import numpy as np
from src.loadSaveModel import saveToPickle, restoreVariables, loadFromPickle

numTrajToSample = 10
maxEpisode = 30000
maxTimeStep = 25
saveInterval = 5000


def main():
    independentVariables = dict()
    # environment parameters
    independentVariables['mapSize'] = [8]
    independentVariables['color'] = [(4, 4), (5, 3), (3, 5)]
    independentVariables['soldiers'] = [(5, 5), (10, 10), (5, 10), (10, 5)]
    # training parameters
    independentVariables['trainEps'] = np.arange(saveInterval, maxEpisode+1, saveInterval)
    independentVariables['evalSequence'] = [(0, 1), (0, 'random'), (1, 'random'), ('random', 0), ('random', 1), ('random', 'random')]
    independentVariables['bufferSize'] = [10000, 100000, 100000]
    independentVariables['minibatchSize'] = [64, 128, 256]
    independentVariables['learnInterval'] = [20, 50, 100]
    independentVariables['layerWidth'] = [32]
    independentVariables['learningRateActor'] = [0.01]
    independentVariables['learningRateCritic'] = [0.01]
    independentVariables['gamma'] = [0.95]
    independentVariables['tau'] = [0.01]

    resultPath = os.path.join(dirName, '..', 'evalResults64')
    # if not os.path.exists(resultPath):
    #     os.makedirs(resultPath)
        
    resultDF = loadFromPickle(os.path.join(resultPath, 'evalResult64.pkl'))

    for evalSequence in independentVariables['evalSequence']:
        resultDF_seq = resultDF.iloc[resultDF.index.get_level_values('evalSequence') == evalSequence]

        for color in independentVariables['color']:
            resultDF_seq_color = resultDF_seq.iloc[resultDF_seq.index.get_level_values('color') == color]

            for soldiers in independentVariables['soldiers']:
                dfToPlot = resultDF_seq_color.iloc[resultDF_seq_color.index.get_level_values('soldiers') == soldiers]

                figure = plt.figure(figsize=(11, 7))
                plotCounter = 1

###### Plot mean total reward + minibatchSize, bufferSize, learnInterval,
                numRows = len(independentVariables['minibatchSize'])
                numColumns = len(independentVariables['bufferSize'])

                for key, outmostSubDf in dfToPlot.groupby('minibatchSize'):
                    print(key)
                    outmostSubDf.index = outmostSubDf.index.droplevel('minibatchSize')
                    for keyCol, outterSubDf in outmostSubDf.groupby('bufferSize'):
                        outterSubDf.index = outterSubDf.index.droplevel('bufferSize')
                        axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
                        for keyRow, innerSubDf in outterSubDf.groupby('learnInterval'):
                            innerSubDf.index = innerSubDf.index.droplevel('learnInterval')
                            innerSubDf.index = innerSubDf.index.droplevel(['color', 'soldiers', 'evalSequence', 'mapSize', 'layerWidth', 'learningRateActor', 'learningRateCritic', 'gamma', 'tau'])
                            plt.ylim([0, 550])

                            innerSubDf.plot.line(ax = axForDraw, y='meanRewardTotal', yerr='seRewardTotal', label = keyRow, uplims=True, lolims=True, capsize=3)

                            if plotCounter <= numColumns:
                                axForDraw.title.set_text('bufferSize = ' + str(keyCol))
                            if plotCounter% numColumns == 1:
                                axForDraw.set_ylabel('minibatchSize = ' + str(key))
                            axForDraw.set_xlabel('epsID')

                        plotCounter += 1
                        # plt.xticks(epsIDList, rotation='vertical')
                        plt.legend(title='learnInterval', title_fontsize = 8, prop={'size': 8})

                figure.text(x=0.03, y=0.5, s='Mean Episode Total Reward', ha='center', va='center', rotation=90)
                plt.suptitle('War total reward: eval {}, color {}, soldier {}'.format(evalSequence, color, soldiers))
                plt.savefig(os.path.join(resultPath, 'evalWar_totalReward_seq{}_color{}_soldier{}' .format(evalSequence, color, soldiers)))
                #

##### Plot mean actions for two agents  + minibatchSize, bufferSize, learnInterval,
                figure = plt.figure(figsize=(11, 11))
                plotCounter = 1

                numRows = len(independentVariables['minibatchSize'])
                numColumns = len(independentVariables['bufferSize'])

                for key, outmostSubDf in dfToPlot.groupby('minibatchSize'):
                    print(key)
                    outmostSubDf.index = outmostSubDf.index.droplevel('minibatchSize')
                    for keyCol, outterSubDf in outmostSubDf.groupby('bufferSize'):
                        outterSubDf.index = outterSubDf.index.droplevel('bufferSize')
                        axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
                        for keyRow, innerSubDf in outterSubDf.groupby('learnInterval'):
                            innerSubDf.index = innerSubDf.index.droplevel('learnInterval')
                            innerSubDf.index = innerSubDf.index.droplevel(['color', 'soldiers', 'evalSequence', 'mapSize', 'layerWidth', 'learningRateActor', 'learningRateCritic', 'gamma', 'tau'])
                            plt.ylim([-1, 8])

                            innerSubDf.plot.line(ax = axForDraw, y='meanActionAgent1', yerr='seActionAgent1', linestyle  = "--", label = str(keyRow) + "Agent1", uplims=True, lolims=True, capsize=3)
                            innerSubDf.plot.line(ax = axForDraw, y='meanActionAgent2', yerr='seActionAgent2', label = str(keyRow)+ "Agent2", uplims=True, lolims=True, capsize=3)
#style=['bs-','ro-','y^-'],
                            # colors = ["blue", "red", "green", "pink", "purple"]
                            # for lineid, line in enumerate(axForDraw.get_lines()):
                            #     colorID = lineid % 3#(len(axForDraw.get_lines())/2)
                            #     line.set_color(colors[int(colorID)])

                            if plotCounter <= numColumns:
                                axForDraw.title.set_text('bufferSize = ' + str(keyCol))
                            if plotCounter% numColumns == 1:
                                axForDraw.set_ylabel('minibatchSize = ' + str(key))
                            axForDraw.set_xlabel('epsID')

                        plotCounter += 1
                        # plt.xticks(epsIDList, rotation='vertical')
                        plt.legend(title='learnInterval', title_fontsize = 8, prop={'size': 8})

                figure.text(x=0.03, y=0.5, s='Mean Agent Action', ha='center', va='center', rotation=90)
                plt.suptitle('War action comparison: eval {}, color {}, soldier {}'.format(evalSequence, color, soldiers))
                plt.savefig(os.path.join(resultPath, 'evalWar_action_seq{}_color{}_soldier{}' .format(evalSequence, color, soldiers)))
                # plt.show()

if __name__ == '__main__':
    main()