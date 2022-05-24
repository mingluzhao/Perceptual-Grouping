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

    conditions['vi385064core2-PowerEdge-R7515'] = {
        'mapSizeLevels': [8],
        'colorLevels': [(4, 4), (5, 3), (3, 5)],
        'maxEpisodeLevels': [30000],
        'maxTimeStepLevels': [25],
        'bufferSizeLevels': [1e4, 1e5],
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