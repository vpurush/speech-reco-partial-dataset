from speechpy import processing as spyproc
# from speechpy import feature as spyfe
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
    # print("frames", fileName, frames.shape, audioData.shape, rate)

    # fft = spyproc.fft_spectrum(frames)
    # fft = numpy.fft.rfft(frames)
    # print("fft", fft[0], numpy.absolute(fft[0]), numpy.angle(fft[0]), fft.shape)

    # for i in range(0, fft.shape[0]):
    #     fftF = fft[i]
    #     var = numpy.var(numpy.absolute(fftF))
    #     print("var", i, var)
    # print("fft 50", fft[50,:])
    # plotSpectrogram(fileName, fft, True, 'fft')

    # exit()

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
    melPowerSpectrogram = libfeature.melspectrogram(audioData, rate, S=None, n_fft=n_fft, hop_length=int(n_fft/2), n_mels=128)
    # print("melSpectrogram", melPowerSpectrogram.shape)
    melDBSpectrogram = librosa.power_to_db(melPowerSpectrogram, ref=numpy.max)
    # print("melDBSpectrogram", melDBSpectrogram.shape)

    plotSpectrogram(fileName, 
                    melDBSpectrogram, 
                    False, 
                    'melSpectrogram')

    # fig = plt.figure(figsize=(15, 4))
    # librosa.display.specshow(librosa.power_to_db(melPowerSpectrogram,
    #                                              ref=numpy.max),
    #                          y_axis='mel', fmax=8000,
    #                          x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel spectrogram')
    # fig.savefig("./output_img/spect/" + fileName + "_melSpectrogram" + "_spect.png")
    # plt.close(fig)


    # chromogram = libfeature.chroma_stft(audioData, rate, n_fft=2048, hop_length=1024)
    # print("chromogram", chromogram, chromogram.shape, numpy.amin(chromogram), numpy.amax(chromogram))
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(chromogram, aspect='auto')
    # ax.set_xticks(numpy.arange(0, chromogram.shape[1], 10))
    # fig.suptitle(fileName)
    # fig.savefig("./output_img/spect/" + fileName + "_chromogram.png")
    # chromogram = numpy.transpose(chromogram)
    
    melDBSpectrogram = numpy.transpose(melDBSpectrogram)
    # print("melDBSpectrogram after transp", melDBSpectrogram.shape)


    return melDBSpectrogram
    # return numpy.transpose(chromogram)

