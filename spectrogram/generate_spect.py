from speechpy import processing as spyproc
from speechpy import feature as spyfe
from matplotlib import pyplot as plt
from librosa import feature as libfeature
import librosa.display
import numpy

def plotSpectrogram(fileName, spectrogram, transposeData, typ):
    if transposeData:
        spectrogram = numpy.transpose(spectrogram)


    fig, ax = plt.subplots(figsize=(10, 4))
    # ax.pcolormesh(numpy.transpose(spectrogram))
    ax.pcolormesh(spectrogram)
    fig.suptitle(fileName)
    fig.savefig("./output_img/spect/" + fileName + "_" + typ + "_spect.png")
    plt.close(fig)

    # fig = plt.figure(figsize=(10, 4))
    # librosa.display.specshow(spectrogram,
    #                          y_axis='mel', fmax=8000,
    #                          x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel spectrogram')
    # fig.savefig("./output_img/spect/" + fileName + "_melSpectrogram" + "_spect.png")
    # plt.close(fig)


def generateSpectrogram(fileName, audioData, rate):
    # return
    # frames = spyproc.stack_frames(audioData, rate, 0.02, 0.02)
    # print("frames", frames, frames.shape)

    # fft = spyproc.fft_spectrum(frames)
    # print("fft", fft, fft.shape)
    # print("fft 50", fft[50,:])
    # plotSpectrogram(fileName, fft, True, 'fft')

    # power = spyproc.power_spectrum(frames)
    # print("power", power, power.shape)
    # plotSpectrogram(fileName, power, True, 'power')


    # logPower = spyproc.log_power_spectrum(frames)
    # print("logPower", logPower, logPower.shape)
    # print("logPower 50", logPower[50,:])
    # plotSpectrogram(fileName, logPower, True, 'logpower')


    # mfccFeatures = spyfe.mfcc(audioData, rate, 0.02, 0.02)
    # print("mfccFeatures", mfccFeatures, mfccFeatures.shape)
    # plotSpectrogram(fileName, mfccFeatures, True, 'mfccFeatures')


    # melsFreqEnergy = spyfe.mfe(audioData, rate, 0.02, 0.02)
    # print("melsFreqEnergy", melsFreqEnergy[0], melsFreqEnergy[0].shape, melsFreqEnergy[1].shape)
    # plotSpectrogram(fileName, melsFreqEnergy[0], True, 'melsFreqEnergy')

    # logMelsFreqEnergy = spyfe.lmfe(audioData, rate, 0.02, 0.02)
    # print("logMelsFreqEnergy", logMelsFreqEnergy, logMelsFreqEnergy.shape)
    # # print("logMelsFreqEnergy 50", logMelsFreqEnergy[10,:], logMelsFreqEnergy[50,:])
    # plotSpectrogram(fileName, logMelsFreqEnergy, True, 'logMelsFreqEnergy')

    # Here n_fft is frame length?
    n_fft = int(0.02 * rate)
    # hop_length=n_fft means no overlap between frames
    melPowerSpectrogram = libfeature.melspectrogram(audioData, rate, S=None, n_fft=n_fft, hop_length=n_fft)
    # print("melSpectrogram", melPowerSpectrogram, melPowerSpectrogram.shape)
    melDBSpectrogram = librosa.power_to_db(melPowerSpectrogram, ref=numpy.max)
    # print("melDBSpectrogram", melDBSpectrogram, melDBSpectrogram.shape)

    # plotSpectrogram(fileName, 
    #                 melDBSpectrogram, 
    #                 False, 
    #                 'melSpectrogram')

    # fig = plt.figure(figsize=(15, 4))
    # librosa.display.specshow(librosa.power_to_db(melPowerSpectrogram,
    #                                              ref=numpy.max),
    #                          y_axis='mel', fmax=8000,
    #                          x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel spectrogram')
    # fig.savefig("./output_img/spect/" + fileName + "_melSpectrogram" + "_spect.png")
    # plt.close(fig)
    
    melDBSpectrogram = numpy.transpose(melDBSpectrogram)
    # print("melDBSpectrogram after transp", melDBSpectrogram, melDBSpectrogram.shape)


    return melDBSpectrogram

