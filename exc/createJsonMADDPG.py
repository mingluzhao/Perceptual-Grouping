import json

def main():
    conditions = dict()

    conditions['Lululucyzs-MacBook-Pro.local'] = {
        'mapSizeLevels': [8],
        'colorLevels': [(5, 3), (3, 5)],
        'maxEpisodeLevels': [20000],
        'maxTimeStepLevels': [25],
        'bufferSizeLevels': [1e4],
        'minibatchSizeLevels': [128],
        'learningRateActorLevels': [0.01],
        'learningRateCriticLevels': [0.01],
        'gammaLevels': [0.95],
        'tauLevels': [0.01],
        'learnIntervalLevels': [100]}

    # conditions['vi385064core2-PowerEdge-R7515'] = { # trained
    #     'mapSizeLevels': [8],
    #     'colorLevels': [(4, 4), (5, 3), (3, 5)],
    #     'maxEpisodeLevels': [30000],
    #     'maxTimeStepLevels': [25],
    #     'bufferSizeLevels': [1e4, 1e5], # ran 1e6 05/28
    #     'minibatchSizeLevels': [64, 128, 256],
    #     'learningRateActorLevels': [0.01],
    #     'learningRateCriticLevels': [0.01],
    #     'gammaLevels': [0.95],
    #     'tauLevels': [0.01],
    #     'learnIntervalLevels': [20, 50, 100]}

    conditions['vi3850-PowerEdge-R7515-2'] = { # then ran one with 64 layer size
        'mapSizeLevels': [8],
        'colorLevels': [(4, 4), (0, 8)],
        'maxEpisodeLevels': [20000],
        'maxTimeStepLevels': [25],
        'bufferSizeLevels': [1e4, 1e5, 1e6],
        'minibatchSizeLevels': [64, 128, 256],
        'learningRateActorLevels': [0.01],
        'learningRateCriticLevels': [0.01],
        'gammaLevels': [0.95],
        'tauLevels': [0.01],
        'learnIntervalLevels': [20, 50, 100]}

    conditions['vi3850-PowerEdge-R7515-1---'] = { # then ran one with 64 layer size
        'mapSizeLevels': [9],
        'colorLevels': [(4, 5), (0, 9)],
        'maxEpisodeLevels': [20000],
        'maxTimeStepLevels': [25],
        'bufferSizeLevels': [1e4, 1e5, 1e6],
        'minibatchSizeLevels': [64, 128, 256],
        'learningRateActorLevels': [0.01],
        'learningRateCriticLevels': [0.01],
        'gammaLevels': [0.95],
        'tauLevels': [0.01],
        'learnIntervalLevels': [20, 50, 100]}

    outputFile = open('conditionsMADDPG.json', 'w')
    json.dump(conditions, outputFile)
    outputFile.close()
    print(conditions.keys())


if __name__ == '__main__':
    main()