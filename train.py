from audio_loader import loadAllFiles;
from spectrogram import generateSpectrogram;

audioDataAndRateArray = loadAllFiles('ch')
print("audioDataAndRateArray", audioDataAndRateArray)

for audioData, rate in audioDataAndRateArray:
    generateSpectrogram(audioData, rate)