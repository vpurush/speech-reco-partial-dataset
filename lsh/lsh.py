import numpy
from lsh import Hasher

class LSH:
    def __init__(self, noOfHashers = 50, noOfHash = 16, dimension = 128):
        # noOfHashers determines the quality of this algorithm
        # noOfHash determines the possible number of buckets a hasher will map a data into
        self.noOfHashers = noOfHashers
        self.hasherList = []
        self.minSupportPercentage = 0.7
        self.minSupportCount = round(self.minSupportPercentage * noOfHashers)

        for i in range(0, noOfHashers):
            hr = Hasher(noOfHash, dimension)
            self.hasherList.append(hr)

    def train(self, data, label):
        for hr in self.hasherList:
            hr.train(data, label)

    def getBucketForData(self, data):
        bucket = []
        for hr in self.hasherList:
            b = hr.getUniqueElementsInBucketForData(data)
            if b is not None:
                bucket.extend(b)

        labelCount = {}
        for label in bucket:
            labelName = label["name"]
            if labelName in labelCount:
                # print("labelCount[labelName]", labelCount[labelName])
                labelCount[labelName][0] += 1
                labelCount[labelName][1].append(label)
            else:
                labelCount[labelName] = [1, [label]]

        # print("labelCount", labelCount)

        bucketWithMinSupport = {}
        for labelName in labelCount:
            if labelCount[labelName][0] > self.minSupportCount:
                bucketWithMinSupport[labelName] = labelCount[labelName][0]

        return bucketWithMinSupport
        # return None


    def __str__(self):
        outputStr = ""

        for hr in self.hasherList:
            outputStr += "\n\n"
            outputStr += str(hr)

        return outputStr

        