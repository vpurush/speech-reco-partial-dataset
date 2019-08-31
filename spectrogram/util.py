import numpy

def padSpectrogramSegment(segment, requiredLength, segmentStart, paddingConstant = -80.):
    # print("segment shape before padding", segment.shape)

    numberOfFreqComponents = segment.shape[1]
    noOfPaddingFramesNeeded = requiredLength - segment.shape[0] - segmentStart
    # print("numberOfFreqComponents", numberOfFreqComponents, noOfPaddingFramesNeeded)

    segmentList = segment.tolist()
    for i in range(0, segmentStart):
        paddingFrame = numpy.repeat(paddingConstant, numberOfFreqComponents)
        segmentList.insert(0, paddingFrame)

    for i in range(0, noOfPaddingFramesNeeded):
        paddingFrame = numpy.repeat(paddingConstant, numberOfFreqComponents)
        segmentList.append(paddingFrame)

    # print("segment shape after padding", numpy.array(segmentList))
    min = numpy.amin(segmentList)
    max = numpy.amax(segmentList)
    # print("segment min max", min, max)


    if min == max and max == paddingConstant:
        return None
    else:
        return numpy.array(segmentList)

def segmentSpectrogram(spectrogram, segmentLength, pading=True):
    # print("spectrogram shape", spectrogram.shape, int(spectrogram.shape[0] / segmentLength))

    segments = []
    i = 0
    while i <= int(spectrogram.shape[0] / segmentLength):
        segmentStart = int(i * segmentLength)
        print("segmentStart", segmentStart)
        segment = spectrogram[segmentStart : segmentStart + segmentLength, :]
        # print("segment inside itr", segment.shape, segment)

        paddedSegment = segment
        if pading:
            paddedSegment = padSpectrogramSegment(segment, spectrogram.shape[0], 0)

        if paddedSegment is not None:
            segments.append(paddedSegment)

        i = i + 0.5

    numpySegments = numpy.array(segments)
    print("segments", numpySegments.shape, numpySegments)
    return numpySegments


def generateTimeShiftedSpectrogram(spectrogram, y, word, k = 10, paddingConstant = -80.):
    totalNoOfFrames = spectrogram.shape[0]

    timeShiftedSpectrograms = []
    timeShiftedSpectrograms.append(numpy.array([spectrogram, y, word]))
    previousSpectrogram = spectrogram

    while True:
        lastKFrames = previousSpectrogram[-1 * k:]
        # print("lastKFrames shape", lastKFrames.shape)
        min = numpy.amin(lastKFrames)
        max = numpy.amax(lastKFrames)

        if(min == max and max == paddingConstant):
            forepart = previousSpectrogram[: -1 * k]
            previousSpectrogram = numpy.append(lastKFrames, forepart, axis = 0)
            # print("previousSpectrogram", previousSpectrogram)
            timeShiftedSpectrograms.append(numpy.array([spectrogram, y, word]))
        else:
            break
    
    return numpy.array(timeShiftedSpectrograms)

def generateTimeShiftedSpectrogramsForArray(data):
    # print("data.shape", data.shape) # -1,3 -> 0 is X, 1 is Y, 2 is word
    timeShiftedSpectrogramsList = None

    for i in range(0, data.shape[0]):
        dataRow = data[i]

        timeShiftedSpectrograms = generateTimeShiftedSpectrogram(dataRow[0], dataRow[1], dataRow[2])
        # print("timeShiftedSpectrograms shape", timeShiftedSpectrograms.shape)

        if timeShiftedSpectrogramsList is None:
            timeShiftedSpectrogramsList = timeShiftedSpectrograms
        else:
            timeShiftedSpectrogramsList = numpy.append(timeShiftedSpectrogramsList, timeShiftedSpectrograms, axis = 0)

    print("timeShiftedSpectrogramsList shape", timeShiftedSpectrogramsList.shape)
    return timeShiftedSpectrogramsList