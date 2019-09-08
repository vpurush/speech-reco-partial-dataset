import numpy

class Hasher:
    def __init__(self, noOfHash, dimension):
        self.noOfHash = noOfHash
        self.dimension = dimension
        self.projection = numpy.random.randn(dimension, noOfHash)
        self.hashTable = dict()

    def convertBoolToInt(self, boolArr):
        output = 0
        for i,j in enumerate(boolArr):
            if j: output += 1<<i
        return output

    def generateHashForData(self, data):
        dotMatrix = numpy.dot(data, self.projection)
        dotMatrix = dotMatrix > 0

        # print("dotMatrix", dotMatrix)
        # exit()

        if dotMatrix.shape[0] > 1:
            raise ValueError("Passing more than one frame?")

        h = self.convertBoolToInt(dotMatrix[0])
        return h

    def train(self, data, label):
        if data.shape[1] != self.dimension:
            raise ValueError("Data does not have proper shape" + str(data.shape))

        h = self.generateHashForData(data)

        if h in self.hashTable:
            self.hashTable[h].append(label)
        else:
            self.hashTable[h] = [label]

    def getBucketForData(self, data):
        h = self.generateHashForData(data)
        if h in self.hashTable:
            return self.hashTable[h]
        else:
            return []
    
    def getUniqueElementsInBucketForData(self, data):
        bucket = self.getBucketForData(data)

        uniqueLabelList = []
        tempLabelNameDict = {}
        for label in bucket:
            if label["name"] not in tempLabelNameDict:
                uniqueLabelList.append(label)
                tempLabelNameDict[label["name"]] = label

        return uniqueLabelList

    def __str__(self):
        outputStr = ""

        hashCount = 0
        for h in self.hashTable:
            hashCount += 1
            outputStr += str(h) + '\n'
            outputStr += str(self.hashTable[h]) + '\n'

        outputStr = "\n Hash Count: " + str(hashCount) + '\n' + outputStr

        return outputStr
        # return ""


        