from os import walk, path
from audio_loader.load_single import loadWavFile

def loadAllFiles(filter):
    output = []
    print('executing loadall files')
    for (dirpath, dirList, fileList) in walk('./recordings'):
        for fileName in fileList:
            if fileName.find(filter) == 0:
                audioWithPadding, rate = loadWavFile(fileName, path.join(dirpath, fileName), True, 3)
                output.append((fileName, audioWithPadding, rate))

    return output