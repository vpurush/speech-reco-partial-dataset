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
    for i in range(0, int(spectrogram.shape[0] / segmentLength) + 1):
        segmentStart = i * segmentLength
        segment = spectrogram[segmentStart : segmentStart + segmentLength, :]
        # print("segment inside itr", segment.shape, segment)

        paddedSegment = segment
        if pading:
            paddedSegment = padSpectrogramSegment(segment, 151, segmentStart)

        if paddedSegment is not None:
            segments.append(paddedSegment)

    numpySegments = numpy.array(segments)
    print("segments", numpySegments.shape, numpySegments)
    return numpySegments
