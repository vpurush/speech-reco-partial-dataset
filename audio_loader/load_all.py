from os import walk, path
from audio_loader.load_single import loadWavFile

def loadAllFiles(folder, filter):
    output = []
    print('executing loadall files')
    for (dirpath, dirList, fileList) in walk('./recordings/' + folder):
        for fileName in fileList:
            if fileName.find(filter) == 0 and fileName.endswith('.wav'):
                audioWithPadding, rate = loadWavFile(fileName, path.join(dirpath, fileName), False, 3, True)
                output.append((fileName, audioWithPadding, rate))

    return output