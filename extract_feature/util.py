import numpy
import random

def generateCombinations(arr, r):
    npArr = numpy.array(arr)

    output = getCombinationForSubArray(npArr, 0, r)
    # print("getCobinationForSubArray output", output)
    return output


def getCombinationForSubArray(arr, position, r):
    print("getCombinationForSubArray start", arr, r)
    output = []
    if position < r:
        for i in range(0, arr.shape[0]):
            combination = numpy.array([arr[i]])
            newArr = arr[[j for j in range(i + 1, arr.shape[0])]]
            
            if position < r - 1:
                childCombinationList = getCombinationForSubArray(newArr, position + 1, r)
                print("childCombinationList", childCombinationList)

                if childCombinationList.shape[0] > 0:
                    for childCombination in childCombinationList:
                        # print("childCombination", childCombination)
                        appendedCombination = numpy.append(combination, childCombination)
                        # print("combination after append", appendedCombination)
                        output.append(appendedCombination)
            else:
                output.append(combination)
                

    return numpy.array(output)

def selectRandomKFromCombinations(arr, kPercent = .2, kMin = 10, kMax = 25, r = 5):
    r = min(r, len(arr))
    print("r", r)
    allPosibleCombinations = generateCombinations(arr, r)
    n = allPosibleCombinations.shape[0]

    k = min(max(int(kPercent * n), kMin), kMax)
    print("allPosibleCombinations.shape[0], k", allPosibleCombinations.shape[0], k)

    if allPosibleCombinations.shape[0] > k:
        kRandomValues = random.sample(range(0, allPosibleCombinations.shape[0]), k)
        # print("kRandomValues", len(kRandomValues))
        return allPosibleCombinations[kRandomValues]
    else:
        return allPosibleCombinations

# Usage
# selectRandomKFromCombinations(range(0, 17))