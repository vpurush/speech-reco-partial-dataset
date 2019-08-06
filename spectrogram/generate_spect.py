from speechpy import processing as spyproc
from speechpy import feature as spyfe
from matplotlib import pyplot as plt
import numpy

def plotSpectrogram(fileName, spectrogram, typ):
    fig, ax = plt.subplots()
    ax.pcolormesh(numpy.transpose(spectrogram))
    fig.suptitle(fileName)
    fig.savefig("./output_img/spect/" + fileName + "_" + typ + "_spect.png")
    plt.close(fig)


def generateSpectrogram(fileName, audioData, rate):
    # return
    frames = spyproc.stack_frames(audioData, rate, 0.02, 0.02)
    print("frames", frames, frames.shape)

    # fft = spyproc.fft_spectrum(frames)
    # print("fft", fft, fft.shape)
    # print("fft 50", fft[50,:])
    # plotSpectrogram(fileName, fft, 'fft')

    # power = spyproc.power_spectrum(frames)
    # print("power", power, power.shape)
    # plotSpectrogram(fileName, power, 'power')


    # logPower = spyproc.log_power_spectrum(frames)
    # print("logPower", logPower, logPower.shape)
    # print("logPower 50", logPower[50,:])
    # plotSpectrogram(fileName, logPower, 'logpower')


    # mfccFeatures = spyfe.mfcc(audioData, rate, 0.02, 0.02)
    # print("mfccFeatures", mfccFeatures, mfccFeatures.shape)
    # plotSpectrogram(fileName, mfccFeatures, 'mfccFeatures')


    # melsFreqEnergy = spyfe.mfe(audioData, rate, 0.02, 0.02)
    # print("melsFreqEnergy", melsFreqEnergy[0], melsFreqEnergy[0].shape, melsFreqEnergy[1].shape)
    # plotSpectrogram(fileName, melsFreqEnergy[0], 'melsFreqEnergy')

    logMelsFreqEnergy = spyfe.lmfe(audioData, rate, 0.02, 0.02)
    print("logMelsFreqEnergy", logMelsFreqEnergy, logMelsFreqEnergy.shape)
    # print("logMelsFreqEnergy 50", logMelsFreqEnergy[10,:], logMelsFreqEnergy[50,:])
    plotSpectrogram(fileName, logMelsFreqEnergy, 'logMelsFreqEnergy')

    return logMelsFreqEnergy

