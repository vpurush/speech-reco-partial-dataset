from audio_loader import loadAllFiles;
from spectrogram import generateSpectrogram;
from cnn import generateOutputVariables, getTwoCharSequencesFromOutput, createModel, trainModel, predict
import numpy

audioDataAndRateArray = loadAllFiles('ch')
print("audioDataAndRateArray", audioDataAndRateArray)

# exit()
trainX = [] # numpy.array([])
trainY = [] # numpy.array([])
for fileName, audioData, rate in audioDataAndRateArray:
    spectrogram = generateSpectrogram(fileName, audioData, rate)

    outputValues = generateOutputVariables(fileName.split('.')[0].split('_')[0])

    trainX.append(spectrogram)
    trainY.append(outputValues)
    # trainX = numpy.append(trainX, [spectrogram])
    # trainY = numpy.append(trainY, [outputValues])

createModel()
trainModel(trainX, trainY)
yPredictions = predict(trainX)

for y in yPredictions:
    getTwoCharSequencesFromOutput(y)