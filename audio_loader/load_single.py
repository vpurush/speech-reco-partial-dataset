from scipy.io import wavfile
import numpy
from matplotlib import pyplot as plt
import librosa
import noisereduce

def loadWavFile(fileName, filePath, savePlot, maxAudioLength, reduceNoise = True):
    # Read file
    # rate, data = wavfile.read(filePath)
    # print(filePath, rate, data.shape, "audio length", data.shape[0] / rate, data[0])

    data, rate = librosa.load(filePath, sr=None)
    # print(filePath, rate, data.shape, "librosa audio length", data.shape[0] / rate, data[0])
    if reduceNoise:
        noiseRemovedData = noisereduce.reduce_noise(audio_clip=data, noise_clip=data[0:10000], verbose=False)
        noiseRemovedData = noisereduce.reduce_noise(audio_clip=noiseRemovedData, noise_clip=data[-10000:], verbose=False)
        data = noiseRemovedData


    maxDataLength = int(maxAudioLength * rate)
    padding = []
    if data.shape[0] > maxDataLength:
        raise ValueError("Max audio length breached")
    else:
        paddingDataLength = maxDataLength - data.shape[0]
        padding = [0 for i in range(paddingDataLength)]

    # data is stereo sound. take left speaker only
    leftSpeakerSound = data # data[:,0]
    # print("leftSpeakerSound.shape", leftSpeakerSound.shape)

    audioWithPadding = numpy.concatenate((leftSpeakerSound, padding))
    # print("audioWithPadding.shape", audioWithPadding.shape)

    if savePlot:
        fig, ax = plt.subplots()
        ax.plot(audioWithPadding)
        fig.suptitle(fileName)
        fig.savefig("./output_img/wav/" + fileName + "_wav.png")
        plt.close(fig)

    return audioWithPadding, rate