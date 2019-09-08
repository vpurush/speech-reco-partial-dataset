from audio_loader import loadAllFiles;
from spectrogram import (generateSpectrogram,
                        )
from extract_feature import (findValidStartEnd,
                            extractValidFrames)

from lsh import (LSH)
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


def loadAudio():
    trainAudioDataAndRateArray = loadAllFiles("train", '')
    testAudioDataAndRateArray = loadAllFiles("test", '')

    return trainAudioDataAndRateArray, testAudioDataAndRateArray


def orchestration():
    trainAudioDataAndRateArray, testAudioDataAndRateArray = loadAudio()

    trainingData = generateData(trainAudioDataAndRateArray)
    testData = generateData(testAudioDataAndRateArray)

    lshObj = LSH()

    for data in trainingData:
        print("fileName", data[1])
        validFrameList = extractValidFrames(data[0])

        for i in range(0, validFrameList.shape[0]):
            reshapedValidFrame = validFrameList[i].reshape(1, -1)
            # print("reshapedValidFrame", reshapedValidFrame)
            lshObj.train(reshapedValidFrame, { "name": data[1] + "_" + str(i), "frameIndex": i })
            # hr.train(validFrames[1:2], data[1])
    
    # print("lshObj", lshObj)

    for testSpect, testFileName in testData:
        print("testSpect.shape", testSpect.shape)
        print("testFileName", testFileName)
        for frame in extractValidFrames(testSpect):
            reshapedValidFrame = frame.reshape(1, -1)
            bucket = lshObj.getBucketForData(reshapedValidFrame)
            print("bucket", bucket)
    # lshObj.getBucketForData(testData[0,0,0])

orchestration()