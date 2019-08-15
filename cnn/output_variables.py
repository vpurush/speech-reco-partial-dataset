import numpy
import math

alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
def generateOutputVariables(word):
    # print('word', word)
    output = numpy.zeros(26 * 26)
    for i in range(1, len(word)):
        twoCharSequence = word[i - 1] + word[i]
        idx = 26 * alphabets.index(word[i - 1]) + alphabets.index(word[i])
        output[idx] = 1
        # print(twoCharSequence, idx, output)

    # output = numpy.append(output, [1, 1])

    # print("generateOutputVariables", output)

    return output


def getTwoCharSequencesFromOutput(output):
    print("getTwoCharSequencesFromOutput", numpy.shape(output))
    indices = [idx for idx in range(0, len(output)) if output[idx] == 1]

    twoCharSequences = []
    for idx in indices:
        firstIndex = math.floor(idx / 26)
        secondIndex = idx % 26

        # print('idx', idx, firstIndex, secondIndex)
        twoCharSequences.append(alphabets[firstIndex] + alphabets[secondIndex])

    
    print('TwoCharSequences', twoCharSequences)


def printOutput(output):
    print("\n\n")
    for idx in range(0, len(output)):
        firstIndex = math.floor(idx / 26)
        secondIndex = idx % 26
        print("chars", alphabets[firstIndex] + alphabets[secondIndex], output[idx])
    