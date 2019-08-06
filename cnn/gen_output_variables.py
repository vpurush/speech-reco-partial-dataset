import numpy

alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
def generateOutputVariables(word):
    output = numpy.zeros(26 * 26)
    for i in range(1, len(word)):
        twoCharSequence = word[i - 1] + word[i]
        idx = 26 * alphabets.index(word[i - 1]) + alphabets.index(word[i])
        output[idx] = 1
        print(twoCharSequence, idx, output)
        getTwoCharSequencesFromOutput(output)


def getTwoCharSequencesFromOutput(output):
    indices = [idx for idx in range(0, len(output)) if output[idx] == 1]
    for idx in indices:
        firstIndex = round(idx / 26)
        secondIndex = idx % 26

        print('TwoCharSequence', alphabets[firstIndex] + alphabets[secondIndex])