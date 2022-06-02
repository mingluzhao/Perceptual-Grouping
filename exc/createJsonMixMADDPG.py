import json

def main():
    conditions = dict()

    conditions['Lululucyzs-MacBook-Pro.local'] = {
        'mapSizeLevels': [8, 9],
        'colorLevels': [(-1, -1)],
        'maxEpisodeLevels': [20000],
        'maxTimeStepLevels': [25],
        'bufferSizeLevels': [1e4],
        'minibatchSizeLevels': [64],
        'learningRateActorLevels': [0.01],
        'learningRateCriticLevels': [0.01],
        'gammaLevels': [0.95],
        'tauLevels': [0.01],
        'learnIntervalLevels': [20],
        'switchIntervalLevels': [200]}

    conditions['vi385064core2-PowerEdge-R7515'] = { # then ran one with 64 layer size
        'mapSizeLevels': [8, 9],
        'colorLevels': [(-1, -1)],
        'maxEpisodeLevels': [20000],
        'maxTimeStepLevels': [25],
        'bufferSizeLevels': [1e4],
        'minibatchSizeLevels': [64],
        'learningRateActorLevels': [0.01],
        'learningRateCriticLevels': [0.01],
        'gammaLevels': [0.95],
        'tauLevels': [0.01],
        'learnIntervalLevels': [20],
        'switchIntervalLevels': [200, 500, 1000]}

    conditions['vi3850-PowerEdge-R7515-1'] = { # then ran one with 64 layer size
        'mapSizeLevels': [9],
        'colorLevels': [(-1, -1)],
        'maxEpisodeLevels': [20000],
        'maxTimeStepLevels': [25],
        'bufferSizeLevels': [1e4, 1e5, 1e6],
        'minibatchSizeLevels': [64, 128, 256],
        'learningRateActorLevels': [0.01],
        'learningRateCriticLevels': [0.01],
        'gammaLevels': [0.95],
        'tauLevels': [0.01],
        'learnIntervalLevels': [20, 50, 100],
        'switchIntervalLevels': list(range(20))}

    outputFile = open('conditionsMixMADDPG.json', 'w')
    json.dump(conditions, outputFile)
    outputFile.close()
    print(conditions.keys())


if __name__ == '__main__':
    main()