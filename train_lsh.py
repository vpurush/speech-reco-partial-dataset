from audio_loader import loadAllFiles;
from spectrogram import (generateSpectrogram,
                        )
from extract_feature import (findValidStartEnd,
                            extractValidFrames)

from lsh import (LSH, 
                findTwoCharSequenceLikelyhood, 
                generateAllPossibleCharSequencesFromLikelyhoodDict,
                sortAllPossibleCharSequenceList,
                findNearestWordList,
                sortNearestWordList)
import numpy
import pickle
import os

def generateData(audioDataAndRateArray):
    dataOutput = []
    for fileName, audioData, rate in audioDataAndRateArray:
        spectrogram = generateSpectrogram(fileName, audioData, rate)

        dataOutput.append(numpy.array([spectrogram, fileName]))

    dataOutput = numpy.array(dataOutput)
    return dataOutput


# def loadAudio():
#     trainAudioDataAndRateArray = loadAllFiles("train", '')
#     testAudioDataAndRateArray = loadAllFiles("test_train", '')

#     return trainAudioDataAndRateArray, testAudioDataAndRateArray

def loadOrTrainLSHModel(forceGenerate = False):
    lshModel = None
    if os.path.exists("./pickle_files/lshModel.pickle") and not forceGenerate:
        print("LSH model found on disk")
        pickleIn = open("./pickle_files/lshModel.pickle","rb")
        lshModel = pickle.load(pickleIn)
    else:
        print("Training LSH model")
        trainAudioDataAndRateArray = loadAllFiles("train", '')
        trainingData = generateData(trainAudioDataAndRateArray)
        
        lshModel = LSH()
        for data in trainingData:
            print("fileName", data[1])
            validFrameList = extractValidFrames(data[0])

            for i in range(0, validFrameList.shape[0]):
                reshapedValidFrame = validFrameList[i].reshape(1, -1)
                # print("reshapedValidFrame", reshapedValidFrame)
                lshModel.train(reshapedValidFrame, { "name": data[1] + "_" + str(i), "frameIndex": i })
                # hr.train(validFrames[1:2], data[1])
        
        # print("lshModel", lshModel)

        pickleOut = open("./pickle_files/lshModel.pickle","wb")
        pickle.dump(lshModel, pickleOut)
        pickleOut.close()

    return lshModel


def orchestration():
    lshObj = loadOrTrainLSHModel(False)

    testAudioDataAndRateArray = loadAllFiles("test_impossible", '')
    testData = generateData(testAudioDataAndRateArray)


    testResults = []
    for testSpect, testFileName in testData:
        # print("\n\n testSpect.shape", testSpect.shape)
        print("\ntestFileName", testFileName)
        bucketList = []
        for frame in extractValidFrames(testSpect):
            reshapedValidFrame = frame.reshape(1, -1)
            bucket = lshObj.getBucketForData(reshapedValidFrame)
            # print("bucket", bucket)
            bucketList.append(bucket)
        
        bucketsOfTwoCharSequences = makePrediction(testFileName, bucketList)
        sortedTwoCharSequenceLikelyhoodList = findTwoCharSequenceLikelyhood(bucketsOfTwoCharSequences)
        allPossibleCharSeq = generateAllPossibleCharSequencesFromLikelyhoodDict(sortedTwoCharSequenceLikelyhoodList)
        sortedAllPossibleCharSeq = sortAllPossibleCharSequenceList(allPossibleCharSeq)
        nearestWordList = findNearestWordList(sortedAllPossibleCharSeq)
        sortedNearstWordList = sortNearestWordList(nearestWordList, sortedAllPossibleCharSeq)
        testResults.append([testFileName, sortedNearstWordList])
        # print("sortedAllPossibleCharSeq", sortedAllPossibleCharSeq)
        # break
    # lshObj.getBucketForData(testData[0,0,0])
    calculateAccuracy(testResults)


def makePrediction(fileName, bucketList, minSupport = 0):
    bucketsOfTwoCharSequences = []
    for bucket in bucketList:
        bucketSize = len(bucket)
        minSupportForBucket = int(minSupport * bucketSize)
        # print("processing new bucket", bucketSize)

        bucketLabelCount = {}
        processedTwoCharSequence = {}

        for labelName in bucket:
            word = labelName.split("_")[0]
            word = word.split(".")[0]

            for i in range(0, len(word) - 1):
                twoCharSequence = word[i] + word[i+1]
                
                if twoCharSequence not in processedTwoCharSequence:
                    processedTwoCharSequence[twoCharSequence] = 1
                else:
                    processedTwoCharSequence[twoCharSequence] += 1

                # print("twoCharSequence", twoCharSequence)

        # twoCharSequencesWithMinSupport = {}
        # for twoCharSequence in processedTwoCharSequence:
        #     if processedTwoCharSequence[twoCharSequence] >= minSupportForBucket:
        #         twoCharSequencesWithMinSupport[twoCharSequence] = processedTwoCharSequence[twoCharSequence]

        # print("twoCharSequencesWithMinSupport", twoCharSequencesWithMinSupport)
        # print("processedTwoCharSequence", fileName, processedTwoCharSequence)
        bucketsOfTwoCharSequences.append(processedTwoCharSequence)
    
    return bucketsOfTwoCharSequences

def calculateAccuracy(testResults):

    matchCount = 0
    top3MatchCount = 0
    top5MatchCount = 0
    for fileName, sortedNearestWordList in testResults:
        
        word = fileName.split("_")[0]
        word = word.split(".")[0]
        # print("word", word)

        if word in sortedNearestWordList:
            index = sortedNearestWordList.index(word)
            # print("inside", index)

            if index < 5:
                top5MatchCount = top5MatchCount + 1
            if index < 3:
                top3MatchCount = top3MatchCount + 1
            if index == 0:
                matchCount = matchCount + 1

    totalNoOfPredictions =  len(testResults)
    print("Accuracy: ", matchCount / totalNoOfPredictions)
    print("Top3 Accuracy ", top3MatchCount / totalNoOfPredictions)
    print("Top5 Accuracy ", top5MatchCount / totalNoOfPredictions)


orchestration()