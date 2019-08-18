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
trainAudioDataAndRateArray = loadAllFiles("train", '')
testAudioDataAndRateArray = loadAllFiles("test", '')
# print("audioDataAndRateArray", audioDataAndRateArray)

# exit()
trainX = [] # numpy.array([])
trainY = [] # numpy.array([])
words = []

def generateData(audioDataAndRateArray):
    dataOutput = []
    for fileName, audioData, rate in audioDataAndRateArray:
        spectrogram = generateSpectrogram(fileName, audioData, rate)

        outputValues = generateOutputVariables(fileName.split('.')[0].split('_')[0])

        trainX.append(spectrogram)
        trainY.append(outputValues)
        words.append(fileName)
        dataOutput.append(numpy.array([spectrogram, outputValues, fileName]))
        # trainX = numpy.append(trainX, [spectrogram])
        # trainY = numpy.append(trainY, [outputValues])

    dataOutput = numpy.array(dataOutput)
    return dataOutput

trainingData = generateData(trainAudioDataAndRateArray)
testData = generateData(testAudioDataAndRateArray)

# createModel()
# trainModel(trainX, trainY)
# yPredictions = predict(trainX)

print("trainingData.shape", trainingData.shape)
print("trainingData[0, :].shape", trainingData[:, 0].shape)

# exit()
fModel = FunctionalModel()
fModel.train(trainingData[:, 0], trainingData[:, 1])
yPredictions = fModel.predict(testData[:, 0])
# exit()

words = testData[:, 2]
i = 0
for y in yPredictions:
    print("word", words[i])
    i = i + 1
    getTwoCharSequencesFromOutput(y)