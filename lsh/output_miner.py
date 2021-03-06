import numpy

def findTwoCharSequenceLikelyhood(bucketsOfTwoCharSequences, vicinityFrameCount = 10):
    twoCharSequenceLikelyhood = {}
    # {
    #   va: [
    #           {
    #                start: 1
    #                likelyhood: 1.2  
    #           }
    #       ]
    # }

    for bucketIdx in range(0, len(bucketsOfTwoCharSequences)):
        bucket = bucketsOfTwoCharSequences[bucketIdx]
        bucketLen = len(bucket)
        # print("processing new bucket")

        if (bucketLen > 0):
            inverseBucketLen = 1 / bucketLen

            for twoCharSequence in bucket:
                # print("twoCharSequence", twoCharSequence)
                vicinityList = []
                if twoCharSequence in twoCharSequenceLikelyhood:
                    vicinityList = twoCharSequenceLikelyhood[twoCharSequence]
                else:
                    twoCharSequenceLikelyhood[twoCharSequence] = vicinityList

                filteredVicinity = filter(lambda x: x["start"] > bucketIdx - vicinityFrameCount, vicinityList)
                # filteredVicinity = vicinityList
                filteredVicinityList = list(filteredVicinity)

                if (len(filteredVicinityList) > 0):
                    vicinity = filteredVicinityList[0]
                    vicinity["likelyhood"] = vicinity["likelyhood"] + inverseBucketLen
                    # print("vicinity found", vicinity)
                else:
                    vicinity = {"start": bucketIdx, "likelyhood": inverseBucketLen}
                    vicinityList.append(vicinity)
                    # print("vicinity not found")

    # print("\n\n twoCharSequenceLikelyhood")
    # for twoCharSequence in twoCharSequenceLikelyhood:
    #     print("twoCharSequence", twoCharSequence, twoCharSequenceLikelyhood[twoCharSequence])

    return twoCharSequenceLikelyhoodDictToList(twoCharSequenceLikelyhood)



def twoCharSequenceLikelyhoodDictToList(twoCharSequenceLikelyhoodDict):
    twoCharSequenceLikelyhoodList = []

    for twoCharSequence in twoCharSequenceLikelyhoodDict:
        vicinityList = twoCharSequenceLikelyhoodDict[twoCharSequence]

        for vicinity in vicinityList:
            vicinity["twoCharSequence"] = twoCharSequence
            twoCharSequenceLikelyhoodList.append(vicinity)

    sortedTwoCharSequenceLikelyhoodList = sorted(twoCharSequenceLikelyhoodList, key=lambda x: x["start"])
    # print("\n\n sortedTwoCharSequenceLikelyhoodList")
    # for vicinity in sortedTwoCharSequenceLikelyhoodList:
    #     print(vicinity)

    return sortedTwoCharSequenceLikelyhoodList

def generateAllPossibleCharSequencesFromLikelyhoodDict(sortedTwoCharSequenceLikelyhoodList, start = -1):
    # print("start", start)
    currentStart = None
    possibleCharSequences = []
    innerPossibleCharSeqList = None

    twoCharLikelyhoodSeqListToBeProcessed = []
    for twoCharSeqLikelyhood in sortedTwoCharSequenceLikelyhoodList:
        # print('twoCharSeqLikelyhood["start"] > start',  twoCharSeqLikelyhood["start"], start, twoCharSeqLikelyhood["start"] > start)
        if ((twoCharSeqLikelyhood["start"] > start) and ((currentStart is None) or (currentStart == twoCharSeqLikelyhood["start"]))):
            currentStart = twoCharSeqLikelyhood["start"]
            twoCharLikelyhoodSeqListToBeProcessed.append(twoCharSeqLikelyhood)

    # print("twoCharLikelyhoodSeqListToBeProcessed")
    # for twoCharSeqLikelyhood in twoCharLikelyhoodSeqListToBeProcessed:
    #     print(twoCharSeqLikelyhood)

    if len(twoCharLikelyhoodSeqListToBeProcessed) > 0:
        innerPossibleCharSeqList = generateAllPossibleCharSequencesFromLikelyhoodDict(sortedTwoCharSequenceLikelyhoodList, currentStart)
        if len(innerPossibleCharSeqList) == 0:
            # print("inner list is empy")
            possibleCharSequences.append({
                "charSequence": "",
                "likelyhood": 0.1
            })
        else:
            # print("inner list is not empy")
            for innerPossibleCharSeq in innerPossibleCharSeqList:
                possibleCharSequences.append({
                    "charSequence": innerPossibleCharSeq["charSequence"],
                    "likelyhood": 0.1 + innerPossibleCharSeq["likelyhood"]
                })

        likelyhoodValueMap = map(lambda x: x["likelyhood"], twoCharLikelyhoodSeqListToBeProcessed)
        maxLikelyhood = max(likelyhoodValueMap)

        for twoCharSeqLikelyhood in twoCharLikelyhoodSeqListToBeProcessed:
            # print('twoCharSeqLikelyhood["start"] > start',  twoCharSeqLikelyhood["start"], start, twoCharSeqLikelyhood["start"] > start)
            if (twoCharSeqLikelyhood["likelyhood"] > 0.5 * maxLikelyhood):
                # print("currentStart", currentStart)

                if len(innerPossibleCharSeqList) == 0:
                    # print("inner list is empy")
                    possibleCharSequences.append({
                        "charSequence": twoCharSeqLikelyhood["twoCharSequence"],
                        "likelyhood": twoCharSeqLikelyhood["likelyhood"]
                    })
                else:
                    # print("inner list is not empy")
                    for innerPossibleCharSeq in innerPossibleCharSeqList:
                        # print("sequences ", innerPossibleCharSeq["charSequence"], )
                        if (len(innerPossibleCharSeq["charSequence"]) == 0):
                            possibleCharSequences.append({
                                "charSequence": twoCharSeqLikelyhood["twoCharSequence"] + innerPossibleCharSeq["charSequence"],
                                "likelyhood": twoCharSeqLikelyhood["likelyhood"] + innerPossibleCharSeq["likelyhood"]
                            })
                        else:
                            if ((twoCharSeqLikelyhood["twoCharSequence"] not in innerPossibleCharSeq["charSequence"]) and
                                (innerPossibleCharSeq["charSequence"][0] == twoCharSeqLikelyhood["twoCharSequence"][1])
                            ):
                                possibleCharSequences.append({
                                    "charSequence": twoCharSeqLikelyhood["twoCharSequence"][0] + innerPossibleCharSeq["charSequence"],
                                    "likelyhood": twoCharSeqLikelyhood["likelyhood"] + innerPossibleCharSeq["likelyhood"]
                                })

    # print("completed", start)

    return possibleCharSequences

def sortAllPossibleCharSequenceList(allPossibleCharSequenceList):
    sortedAllPossibleCharSeqList = sorted(allPossibleCharSequenceList, key = lambda x: x["likelyhood"], reverse=True)

    print("\n sortAllPossibleCharSequenceList")
    for itm in sortedAllPossibleCharSeqList:
        print(itm)
    
    mappedSortedAllPossibleCharSeqList = map(lambda x: x["charSequence"], sortedAllPossibleCharSeqList)
    return list(mappedSortedAllPossibleCharSeqList)


# def sortNearestWordList(nearestWordList, sortedAllPossibleCharSeq):
#     # print("aa", nearestWordList, sortedAllPossibleCharSeq)
#     firstCharLikelyhood = {}
#     for i in range(0, min(10, len(sortedAllPossibleCharSeq))):
#         word = sortedAllPossibleCharSeq[i]
#         firstChar = word[0]
        
#         if firstChar in firstCharLikelyhood:
#             firstCharLikelyhood[firstChar] = firstCharLikelyhood[firstChar] + 1
#         else:
#             firstCharLikelyhood[firstChar] = 1

#     sortedNearestWordList = sorted(nearestWordList, key=lambda x: -1 * firstCharLikelyhood[x[0]] if x[0] in firstCharLikelyhood else 100)
#     # print("sortedNearestWordList", firstCharLikelyhood, nearestWordList, sortedNearestWordList)
#     print("sortedNearestWordList", sortedNearestWordList)
#     return sortedNearestWordList


def sortLogic(sortedAllPossibleCharSeq):
    first10PossibleCharSeq = sortedAllPossibleCharSeq[0:9]
    def sortMethod(x):
        score = 0
        for charSeq in first10PossibleCharSeq:
            if x in charSeq:
                score = score + 10
            else:
                for xChar in x:
                    if xChar in charSeq:
                        score = score + 1

        return score

    return sortMethod
    
def sortNearestWordList(nearestWordList, sortedAllPossibleCharSeq):
    # print("aa", nearestWordList, sortedAllPossibleCharSeq)
    firstCharLikelyhood = {}
    for i in range(0, min(10, len(sortedAllPossibleCharSeq))):
        word = sortedAllPossibleCharSeq[i]
        firstChar = word[0]
        
        if firstChar in firstCharLikelyhood:
            firstCharLikelyhood[firstChar] = firstCharLikelyhood[firstChar] + 1
        else:
            firstCharLikelyhood[firstChar] = 1

    sortedNearestWordList = sorted(nearestWordList, key=sortLogic(sortedAllPossibleCharSeq), reverse=True)
    # print("sortedNearestWordList", firstCharLikelyhood, nearestWordList, sortedNearestWordList)
    print("sortedNearestWordList", sortedNearestWordList)
    return sortedNearestWordList



