import numpy

def compareTwoSpect(spect1, spect2, frameCount = 4):
    (start1, end1) = findValidStartEnd(spect1)
    (start2, end2) = findValidStartEnd(spect2)

    print("start end", start1, end1, start2, end2, spect1.shape)

    comparisonList = []
    for i in range(start1, end1 - frameCount + 1):
        spect1NFrames = spect1[i : i + frameCount]
        # print("spect1NFrames", spect1NFrames.shape)

        for j in range(start2, end2 - frameCount + 1):
            spect2NFrames = spect2[j : j + frameCount]
            # print("spect2NFrames", spect2NFrames.shape)

            comparisonScore = compareFrames(spect1NFrames, spect2NFrames)
            comparisonList.append((comparisonScore, i, j))

    # print("comparisonList", comparisonList)
    comparisonList = comparisonList[0:10]
    sortedComparisonList = sorted(comparisonList, key=lambda x: x[0])
    print("sortedComparisonList", sortedComparisonList)

    return sortedComparisonList


def getComparisonMatrix(spectArray):
    comparisonMatrix = {}
    for i in range(0, spectArray.shape[0]):
        for j in range(i + 1, spectArray.shape[0]):
            if i != j:
                comparison = compareTwoSpect(spectArray[i], spectArray[j])
                comparisonInReverse = compareTwoSpect(spectArray[j], spectArray[i])
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
                            print("innerMasterComparison", innerMasterComparison)
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
    print("sortedMasterComparisonList", index, sortedMasterComparisonList, clonedVisited)
    return sortedMasterComparisonList
                    



def startSpectComparison(spectArray):
    # print("spectArray.shape", spectArray.shape)
    if spectArray.shape[0] > 1:
        comparisonMatrix = getComparisonMatrix(spectArray)
        findBestPair(spectArray.shape[0], comparisonMatrix)

def frameContainsValidInfo(frame):
    var = numpy.var(frame)
    print("var", var)
    return var > 250

def findValidStartEnd(spect):
    start = -1
    end = -1
    for i in range(0, spect.shape[0]):
        if (frameContainsValidInfo(spect[i, :])):
            start = i
            break

    for j in reversed(range(0, spect.shape[0])):
        if (frameContainsValidInfo(spect[j, :])):
            end = j
            break

    return (start, end)




def compareFrames(spect1NFrames, spect2NFrames):
    diff = spect1NFrames - spect2NFrames
    absDiff = abs(diff)
    mean = numpy.mean(absDiff)
    
    return mean