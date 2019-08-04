from audio_loader import loadAllFiles;
from spectrogram import generateSpectrogram;

audioDataAndRateArray = loadAllFiles('va')
print("audioDataAndRateArray", audioDataAndRateArray)

for fileName, audioData, rate in audioDataAndRateArray:
    generateSpectrogram(fileName, audioData, rate)