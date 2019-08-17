from audio_loader import loadAllFiles;
from spectrogram import generateSpectrogram;
from cnn import (generateOutputVariables, 
                getTwoCharSequencesFromOutput, 
                createModel, 
                trainModel,
                predict, 
                normalizeData, 
                FunctionalModel)
import numpy

# normalizeData(numpy.array([
#     numpy.array([2, 3, 4, 5]),
#     numpy.array([7, 8, 9, 10])
# ]))

# exit()
audioDataAndRateArray = loadAllFiles('')
# print("audioDataAndRateArray", audioDataAndRateArray)

# exit()
trainX = [] # numpy.array([])
trainY = [] # numpy.array([])
words = []
for fileName, audioData, rate in audioDataAndRateArray:
    spectrogram = generateSpectrogram(fileName, audioData, rate)

    outputValues = generateOutputVariables(fileName.split('.')[0].split('_')[0])

    trainX.append(spectrogram)
    trainY.append(outputValues)
    words.append(fileName)
    # trainX = numpy.append(trainX, [spectrogram])
    # trainY = numpy.append(trainY, [outputValues])

# createModel()
# trainModel(trainX, trainY)
# yPredictions = predict(trainX)

fModel = FunctionalModel()
fModel.train(trainX, trainY)
yPredictions = fModel.predict(trainX)
# exit()

i = 0
for y in yPredictions:
    print("word", words[i])
    i = i + 1
    getTwoCharSequencesFromOutput(y)