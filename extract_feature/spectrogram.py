import numpy
from extract_feature import selectRandomKFromCombinations
from matplotlib import pyplot as plt

def compareTwoSpect(spect1, spect2, frameCount):
    (start1, end1) = findValidStartEnd(spect1)
    (start2, end2) = findValidStartEnd(spect2)

    # print("start end", start1, end1, start2, end2, spect1.shape)

    comparisonList = []
    for i in range(start1, end1 - frameCount + 1):
        # print("i", i)
        spect1NFrames = spect1[i : i + frameCount]
        # print("spect1NFrames", spect1NFrames.shape)

        for j in range(start2, end2 - frameCount + 1):
            spect2NFrames = spect2[j : j + frameCount]
            # print("spect2NFrames", spect2NFrames.shape)

            comparisonScore = compareFrames(spect1NFrames, spect2NFrames)
            comparisonList.append((comparisonScore, i, j))

        # if i > start1:
        #     exit()

    # print("comparisonList", comparisonList)
    comparisonList = comparisonList[0:3]
    sortedComparisonList = sorted(comparisonList, key=lambda x: x[0])
    print("sortedComparisonList", sortedComparisonList)

    return sortedComparisonList


def getComparisonMatrix(spectArray, frameCount):
    comparisonMatrix = {}
    for i in range(0, spectArray.shape[0]):
        for j in range(i + 1, spectArray.shape[0]):
            if i != j:
                comparison = compareTwoSpect(spectArray[i], spectArray[j], frameCount)
                comparisonInReverse = compareTwoSpect(spectArray[j], spectArray[i], frameCount)
                if i not in comparisonMatrix:
                    comparisonMatrix[i] = {}
                if j not in comparisonMatrix:
                    comparisonMatrix[j] = {}

                comparisonMatrix[i][j] = comparison
                comparisonMatrix[j][i] = comparisonInReverse

    print("comparisonMatrix", comparisonMatrix)
    return comparisonMatrix

def findBestPair(spectArrayLen, comparisonMatrix, index = 0, framePositionList = None, visited = []):
    if framePositionList is None:
        framePositionList = [None for i in range(0, spectArrayLen)]
    
    masterComparisonList = []
    clonedVisited = visited[:]
    clonedVisited.append(index)
    if index in comparisonMatrix:
        comparisonRow = comparisonMatrix[index]
        framePositionLeft = framePositionList[index]
        for key in comparisonRow:
            if key not in clonedVisited:
                framePositionRight = framePositionList[key]
                if framePositionLeft == None:
                    comparisonList = comparisonRow[key]
                else:
                    comparisonList = comparisonRow[key]

                    if framePositionRight is None:
                        comparisonList = [itm for itm in comparisonList if itm[1] == framePositionLeft]
                    else:
                        comparisonList = [itm for itm in comparisonList if itm[1] == framePositionLeft and itm[2] == framePositionRight]


                if len(comparisonList) > 0:
                    for comparison in comparisonList:
                        clonedFramePositionList = framePositionList[:]
                        clonedFramePositionList[key] = comparison[2]

                        innerMasterComparisonList = findBestPair(spectArrayLen, comparisonMatrix, key, clonedFramePositionList, clonedVisited)
                        for innerMasterComparison in innerMasterComparisonList:
                            # print("innerMasterComparison", innerMasterComparison)
                            comparisonScore = comparison[0] + innerMasterComparison[0]
                            masterIndexList = [comparison[1]]
                            for innerMasterIndex in innerMasterComparison[1]:
                                masterIndexList.append(innerMasterIndex)

                            masterComparisonList.append((comparisonScore, masterIndexList, innerMasterComparison[2]))
                else:
                    # Frame not found while comparing with next spectrogram
                    masterComparisonList.append((9999., [framePositionLeft], clonedVisited))

    if len(masterComparisonList) == 0:
        masterComparisonList.append((0., [framePositionLeft], clonedVisited))

    sortedMasterComparisonList = sorted(masterComparisonList, key=lambda x: x[0])
    # print("sortedMasterComparisonList", len(sortedMasterComparisonList))
    return sortedMasterComparisonList
                    

def findBestBatch(spectArray, frameCount):
    randomKCombinationList = selectRandomKFromCombinations(range(0, spectArray.shape[0]))
    print("randomKCombinationList", randomKCombinationList, spectArray.shape[0])
    batchList = []

    for combi in randomKCombinationList:
        subsetKSpectArray = spectArray[combi]

        comparisonMatrix = getComparisonMatrix(subsetKSpectArray, frameCount)

        sortedMasterComparisonList = findBestPair(subsetKSpectArray.shape[0], comparisonMatrix)
        bestPair = sortedMasterComparisonList[0]
        (bestPairComparisonScore, bestPairStartIndices, bestPairSpectIndices) = bestPair
        # print("bestPair", bestPair, bestPairComparisonScore, bestPairStartIndices, bestPairSpectIndices)

        nFramesList = []
        for i in range(0, len(bestPairSpectIndices)):
            spectIndex = bestPairSpectIndices[i]
            startIndex = bestPairStartIndices[i]
            # print("spectIndex", spectIndex)
            nFrames = subsetKSpectArray[spectIndex][startIndex: startIndex + frameCount]
            nFramesList.append(nFrames)
            # print("nFrames.shape", nFrames.shape)

        nFramesList = numpy.array(nFramesList)
        # print("nFramesList.shape", nFramesList.shape)

        batchList.append(numpy.array([bestPairComparisonScore, nFramesList]))

    sortedBatchList = sorted(batchList, key=lambda x: x[0])
    sortedBatchList = numpy.array(sortedBatchList)
    # print("sortedBatchList", sortedBatchList, sortedBatchList.shape)

    output = numpy.array(sortedBatchList[:,1])
    # print("output", output, output.shape)

    return output
        


def getSpectNFrames(spectArray, frameCount = 4):
    # print("spectArray.shape", spectArray.shape)

    return findBestBatch(spectArray, frameCount)

def frameContainsValidInfo(frame):
    var = numpy.var(frame)
    # print("var", var)
    return var > 100
    # return var > 0.02

def findValidStartEnd(spect):
    start = -1
    end = -1
    for i in range(0, spect.shape[0]):
        # print("i", i)
        if (frameContainsValidInfo(spect[i, :]) and 
            frameContainsValidInfo(spect[i + 1, :]) and 
            frameContainsValidInfo(spect[i + 2, :])):
            start = i
            break

    for j in reversed(range(0, spect.shape[0])):
        # print("j", j)
        if (frameContainsValidInfo(spect[j, :]) and 
            frameContainsValidInfo(spect[j - 1, :]) and 
            frameContainsValidInfo(spect[j - 2, :])):
            end = j
            break

    if end-start > 70:
        print("\n \n ----- Possible incorrect indentification of valid frames", end-start)

    print("valid start end", start, end)
    return (start, end)

def extractValidFrames(spect):
    start, end = findValidStartEnd(spect)
    validFrames = spect[start : end + 1]
    print("validFrames", validFrames.shape)
    return validFrames

    # validFrames = []
    # for frame in spect:
    #     if(frameContainsValidInfo(frame)):
    #         validFrames.append(frame)

    # validFrames = numpy.array(validFrames)
    # print("validFrames.shape", validFrames.shape)
    # return validFrames




def compareFrames(spect1NFrames, spect2NFrames):
    diff = spect1NFrames - spect2NFrames
    absDiff = abs(diff)
    # mean = numpy.mean(absDiff)
    # mx = 0
    # for diff in absDiff:
    #     mx = max()
    # plt.imshow(spect1NFrames)
    # plt.show()
    # plt.imshow(spect2NFrames)
    # plt.show()
    # maxMean = max(numpy.mean(absDiff, axis = 1))
    # countLargeDiff = numpy.sum(absDiff > 1.5)
    # expSum = numpy.sum(numpy.exp(absDiff))
    twoPowSum = numpy.sum(numpy.power(1.2, absDiff))
    
    # print("absDiff", absDiff.shape, mean, maxMean, countLargeDiff, expSum, twoPowSum)
    return twoPowSum