import numpy
import librosa
from matplotlib import pyplot as plt
import scipy
import noisereduce

def plotFFTSpectrogram():
    
    data, rate = librosa.load('./recordings/_blank.wav', sr=None)

    fftOutput = numpy.fft.fft(data)
    print("fftOutput", fftOutput, fftOutput.shape, len(data), rate)
    print("fftOutput.real", fftOutput.real)
    print("fftOutput.imag", fftOutput.imag)
    sfft = scipy.fftpack.fft(data)
    print("scipy.fftpack.fft", sfft, sfft.shape)
    # plotImage(data, "_blank_wav")

    fig, ax = plt.subplots(1, 1)
    ax.plot(numpy.arange(len(data)),fftOutput.real)
    ax.set_xlabel('Freq')
    ax.set_ylabel('Amplitude')

    plt.show()


def plotImage(data, fileName):
    plt.imsave("./output_img/fft/" + fileName + ".png", data)


# plotFFTSpectrogram()

def showWave(data, fileName):
    
    fig, ax = plt.subplots()
    ax.plot(data)
    fig.suptitle(fileName)
    plt.show()
    # fig.savefig("./output_img/wav/" + fileName + "_wav.png")
    # plt.close(fig)


def reduceNoise():
    blankData, blankDataRate = librosa.load('./recordings/_blank.wav', sr=None)
    aimData, aimDataRate = librosa.load('./recordings/aim.wav', sr=None)

    noiseData = aimData[0:10000]

    showWave(aimData, "before")

    noiseRemovedData = noisereduce.reduce_noise(audio_clip=aimData, noise_clip=noiseData, verbose=False)
    showWave(noiseRemovedData, "after")


    print("aimData", aimData.shape, aimDataRate)


reduceNoise()
