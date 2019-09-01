from audio_loader import loadAllFiles;
from spectrogram import (generateSpectrogram,
                        segmentSpectrogram,
                        generateTimeShiftedSpectrogram,
                        generateTimeShiftedSpectrogramsForArray)
from cnn import (generateOutputVariables, 
                getTwoCharSequencesFromOutput, 
                createModel, 
                trainModel,
                predict, 
                normalizeData, 
                FunctionalModel)
from extract_feature import (compareTwoSpect,
                            getComparisonMatrix,
                            getSpectNFrames)
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


def generateTrainingDataForDTree(tDataList):
    twoCharSequenceList = ['ai', 'ch']
    trainingDataDict = {}
    for twoCharSequence in twoCharSequenceList:
        dataForTwoCharSequence = []
        for tData in tDataList:
            (spectrogram, outputValues, fileName) = tData
            if(fileName.find(twoCharSequence) != -1):
                dataForTwoCharSequence.append(spectrogram)

        trainingDataDict[twoCharSequence] = numpy.array(dataForTwoCharSequence)

    return trainingDataDict

trainingDataDict = generateTrainingDataForDTree(trainingData)

for key in trainingDataDict:
    print("shape:" + key, trainingDataDict[key].shape)
print("trainingDataDict", trainingDataDict)

exit()

nFramesList = getSpectNFrames(trainingData[:, 0], 6)
print("nFramesList.shape", nFramesList.shape)

print("training words", trainingData[:, 2])
exit()
# createModel()
# trainModel(trainX, trainY)
# yPredictions = predict(trainX)

# print("trainingData.shape", trainingData.shape)
# print("trainingData[0, :].shape", trainingData[:, 0].shape)

# tempFirstSpectrogram = testData[:, 0][3]
timeShiftedTrainingData = generateTimeShiftedSpectrogramsForArray(trainingData)
# print("spectrogram", tempFirstSpectrogram)
# segmentedSpectrogram = segmentSpectrogram(tempFirstSpectrogram, 151)

fModel = FunctionalModel()
fModel.trainOrRestore(trainingData[:, 0], trainingData[:, 1], True)
# fModel.trainOrRestore(timeShiftedTrainingData[:, 0], timeShiftedTrainingData[:, 1], False)
# fModel.trainOrRestore(segmentedSpectrogram, trainingData[:, 1], True)
yPredictions = fModel.predict(testData[:, 0])
# yPredictions = fModel.predict(segmentedSpectrogram)
# exit()


# print("training words", trainingData[:, 2])
print("testing words", testData[:, 2])
words = testData[:, 2]
i = 0
for y in yPredictions:
    # print("word", words[i])
    i = i + 1
    getTwoCharSequencesFromOutput(y)