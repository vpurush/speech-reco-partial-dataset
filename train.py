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
                FunctionalModel,
                NFramesModel)
from extract_feature import (compareTwoSpect,
                            getComparisonMatrix,
                            findBestBatch,
                            performPCA,
                            findValidStartEnd,
                            transformUsingPCA)
from dtree import (trainDTreeClassifier,
                    printDTreeClassifiers,
                    predictDTree)
import numpy
import pickle
import os

# normalizeData(numpy.array([
#     numpy.array([2, 3, 4, 5]),
#     numpy.array([7, 8, 9, 10])
# ]))

# exit()
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

# trainingData = generateData(trainAudioDataAndRateArray)
# testData = generateData(testAudioDataAndRateArray)
trainingData = None
testData = None


def generateTrainingDataDictionary(tDataList):
    twoCharSequenceList = ['ai', 'ch']
    trainingDataDict = {}
    for twoCharSequence in twoCharSequenceList:
        print("processsing", twoCharSequence)
        dataForTwoCharSequence = []
        for tData in tDataList:
            (spectrogram, outputValues, fileName) = tData
            if(fileName.find(twoCharSequence) != -1):
                dataForTwoCharSequence.append(spectrogram)

        trainingDataDict[twoCharSequence] = numpy.array(dataForTwoCharSequence)

    return trainingDataDict

def generateMergedBatchTrainingData(tDataList):
    mBatchTrainingData = {}
    trainingDataDict = generateTrainingDataDictionary(tDataList)
    for key in trainingDataDict:
        # print("shape:" + key, trainingDataDict[key].shape)
        mergedBatchList = generateBatchAndMerge(trainingDataDict[key])
        mBatchTrainingData[key] = mergedBatchList

    return mBatchTrainingData

def generateBatchAndMerge(trainingDataList):
    batchList = findBestBatch(trainingDataList, 4)
    # print("batchList.shape", batchList.shape)

    mergedBatchList = None
    for batch in batchList:
        if mergedBatchList is None:
            mergedBatchList = batch
        else:
            mergedBatchList = numpy.append(mergedBatchList, batch, axis = 0)

    print("mergedBatchList", mergedBatchList.shape)
    return mergedBatchList


def flattenNFrameList(nFrameList):
    
    flattenedNFrameList = []
    for nFrame in nFrameList:
        flattenedNFrameList.append(nFrame.flatten())

    return numpy.array(flattenedNFrameList)


def generateFlattenedTDataDict(tDataList):
    flattenedTDataDict = {}
    mBatchTrainingData = generateMergedBatchTrainingData(trainingData)
    for key in mBatchTrainingData:
        nFrameList = mBatchTrainingData[key]
        print("nFrameList.shape", key, nFrameList.shape)

        flattenedNFrameList = flattenNFrameList(nFrameList)
        print("flattenedNFrameList", flattenedNFrameList.shape)

        flattenedTDataDict[key] = flattenedNFrameList

    return flattenedTDataDict



def performPCAOnFlattenedTData(tDataList):
    tDataAfterPCA = {}
    flattenedTDataDict = generateFlattenedTDataDict(tDataList)

    return flattenedTDataDict
    
    # for key in flattenedTDataDict:
    #     tDataAfterPCA[key] = performPCA(key, flattenedTDataDict[key])

    # return tDataAfterPCA

def genDataWithPandN(tDataList):
    tDataWithPandN = {}
    tDataAfterPCA = performPCAOnFlattenedTData(tDataList)

    for key in tDataAfterPCA:
        positiveInput = tDataAfterPCA[key]
        negativeInput = None
        for innerKey in tDataAfterPCA:
            if innerKey != key:
                if negativeInput is None:
                    negativeInput = tDataAfterPCA[innerKey]
                else:
                    negativeInput = numpy.append(negativeInput, tDataAfterPCA[innerKey], axis = 0)


        positiveOutput = numpy.full((positiveInput.shape[0]), 1)
        negativeOutput = numpy.full((negativeInput.shape[0]), 0)

        inp = numpy.append(positiveInput, negativeInput, axis = 0)
        out = numpy.append(positiveOutput, negativeOutput, axis = 0)

        print("inp.shape, out.shape", inp.shape, out.shape)

        tDataWithPandN[key] = (inp, out)
        # trainDTreeClassifier(key, inp, out)

    # print("tDataWithPandN", tDataWithPandN)

    return tDataWithPandN

tDataWithPandN = None
def readTDataWithPandNorGenerate(forceGenerate = False):
    global tDataWithPandN, trainingData

    if os.path.exists("./pickle_files/tDataWithPandN.pickle") and not forceGenerate:
        print("tDataWithPandN found on disk")
        pickleIn = open("./pickle_files/tDataWithPandN.pickle","rb")
        tDataWithPandN = pickle.load(pickleIn)
    else:
        print("tDataWithPandN not found on disk. Generating")
        trainAudioDataAndRateArray = loadAllFiles("train", '')
        testAudioDataAndRateArray = loadAllFiles("test", '')
        trainingData = generateData(trainAudioDataAndRateArray)
        testData = generateData(testAudioDataAndRateArray)
        tDataWithPandN = genDataWithPandN(trainingData)
        pickleOut = open("./pickle_files/tDataWithPandN.pickle","wb")
        pickle.dump(tDataWithPandN, pickleOut)
        pickleOut.close()
# printDTreeClassifiers()

readTDataWithPandNorGenerate(True)


key = 'ch'
nFramesModel = NFramesModel(key)
nFramesModel.trainOrRestore(tDataWithPandN[key][0], tDataWithPandN[key][1], True)

def genTestData(key, segmentedSpectrogram):
    flattenedNFrameList = flattenNFrameList(segmentedSpectrogram)
    print("flattenedNFrameList.shape", flattenedNFrameList.shape)
    return transformUsingPCA('ai', flattenedNFrameList)

print("testing words", testData[:, 2])
tempFirstSpectrogram = testData[:, 0][5]
(vStart, vEnd) = findValidStartEnd(tempFirstSpectrogram)
tempFirstSpectrogram = tempFirstSpectrogram[vStart: vEnd + 1]
print("tempFirstSpectrogram", tempFirstSpectrogram.shape)
segmentedSpectrogram = segmentSpectrogram(tempFirstSpectrogram, 6, False)



testData = genTestData('ai', segmentedSpectrogram)
print("testData.shape", testData.shape)
predictDTree('ai', testData)
testData = genTestData('ch', segmentedSpectrogram)
print("testData.shape", testData.shape)
predictDTree('ch', testData)


# for key in trainingDataDict:
#     print("shape:" + key, trainingDataDict[key].shape)
# print("trainingDataDict", trainingDataDict)

exit()


print("training words", trainingData[:, 2])
exit()
# createModel()
# trainModel(trainX, trainY)
# yPredictions = predict(trainX)

# print("trainingData.shape", trainingData.shape)
# print("trainingData[0, :].shape", trainingData[:, 0].shape)

# tempFirstSpectrogram = testData[:, 0][3]
# timeShiftedTrainingData = generateTimeShiftedSpectrogramsForArray(trainingData)
# print("spectrogram", tempFirstSpectrogram)
# segmentedSpectrogram = segmentSpectrogram(tempFirstSpectrogram, 151)

# fModel = FunctionalModel()
# fModel.trainOrRestore(trainingData[:, 0], trainingData[:, 1], True)
# fModel.trainOrRestore(timeShiftedTrainingData[:, 0], timeShiftedTrainingData[:, 1], False)
# fModel.trainOrRestore(segmentedSpectrogram, trainingData[:, 1], True)
# yPredictions = fModel.predict(testData[:, 0])
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