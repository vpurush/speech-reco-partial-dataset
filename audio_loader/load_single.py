from scipy.io import wavfile
import numpy
from matplotlib import pyplot as plt

def loadWavFile(fileName, filePath, savePlot, maxAudioLength):
    # Read file
    rate, data = wavfile.read(filePath)
    print(filePath, rate, data.shape, "audio length", data.shape[0] / rate)

    maxDataLength = int(maxAudioLength * rate)
    padding = []
    if data.shape[0] > maxDataLength:
        raise ValueError("Max audio length breached")
    else:
        paddingDataLength = maxDataLength - data.shape[0]
        padding = [0 for i in range(paddingDataLength)]

    # data is stereo sound. take left speaker only
    leftSpeakerSound = data[:,0]
    print("leftSpeakerSound.shape", leftSpeakerSound.shape)

    audioWithPadding = numpy.concatenate((leftSpeakerSound, padding))
    print("audioWithPadding.shape", audioWithPadding.shape)

    if savePlot:
        fig, ax = plt.subplots()
        ax.plot(audioWithPadding)
        fig.suptitle(fileName)
        fig.savefig("./output_img/wav/" + fileName + "_wav.png")

    return audioWithPadding, rate