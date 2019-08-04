from speechpy import processing as spyproc
from speechpy import feature as spyfe

def generateSpectrogram(audioData, rate):
    # return
    frames = spyproc.stack_frames(audioData, rate, 0.03, 0.03)
    print("frames", frames, frames.shape)

    fft = spyproc.fft_spectrum(frames)
    print("fft", fft, fft.shape)

    power = spyproc.power_spectrum(frames)
    print("power", power, power.shape)

    logPower = spyproc.log_power_spectrum(frames)
    print("logPower", logPower, logPower.shape)

    mfccFeatures = spyfe.mfcc(audioData, rate, 0.02, 0.02)
    print("mfccFeatures", mfccFeatures, mfccFeatures.shape)

    melsFreqEnergy = spyfe.mfe(audioData, rate, 0.02, 0.02)
    print("melsFreqEnergy", melsFreqEnergy, melsFreqEnergy[0].shape, melsFreqEnergy[1].shape)