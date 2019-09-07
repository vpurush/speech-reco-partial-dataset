import numpy
from lsh import Hasher

class LSH:
    def __init__(self, noOfHashers = 5, noOfHash = 15, dimension = 12):
        self.noOfHashers = noOfHashers
        self.hasherList = []

        for i in range(0, noOfHashers):
            hr = Hasher(noOfHash, dimension)
            self.hasherList.append(hr)

    def train(self, data, label):
        for hr in self.hasherList:
            hr.train(data, label)

    def __str__(self):
        outputStr = ""

        for hr in self.hasherList:
            outputStr += "\n\n"
            outputStr += str(hr)

        return outputStr

        