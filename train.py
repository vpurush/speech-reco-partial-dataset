from audio_loader import loadAllFiles;
from spectrogram import generateSpectrogram;
from cnn import generateOutputVariables

audioDataAndRateArray = loadAllFiles('chain')
print("audioDataAndRateArray", audioDataAndRateArray)

for fileName, audioData, rate in audioDataAndRateArray:
    spectrogram = generateSpectrogram(fileName, audioData, rate)

    generateOutputVariables(fileName.split('.')[0])