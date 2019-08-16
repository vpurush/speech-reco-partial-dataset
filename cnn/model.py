from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, AveragePooling1D
from keras.utils import normalize, plot_model
from datetime import datetime
from spectrogram import plotSpectrogram
from cnn import printOutput
import tensorflow as tf
import keras
import numpy

model = None
def createModel():
    global model
    if(model == None):

        model = Sequential()
        conv1 = Conv1D(filters=20, kernel_size=3, strides=1, use_bias=False, activation=None, input_shape=(151,))
        model.add(conv1)
        pool1 = AveragePooling1D(pool_size=3, strides=3)
        model.add(pool1)

        print("shape conv1", conv1.output_shape)
        print("shape pool1", pool1.output_shape)
        # model.add(Conv1D(filters=10, kernel_size=4, strides=1, use_bias=False, activation=None))
        # model.add(AveragePooling1D(pool_size=2, strides=2))
        # model.add(Conv1D(filters=10, kernel_size=4, strides=1, use_bias=False, activation=None))
        # model.add(AveragePooling1D(pool_size=2, strides=2))

        model.add(Flatten())
        # model.add(Dense(units=(26*26 + 2) * 2, activation='relu'))
        leakyReluActivation = keras.layers.LeakyReLU(alpha=0.3)
        model.add(Dense(units=26*26 * 4, activation=leakyReluActivation))
        model.add(Dense(units=26*26, activation='sigmoid'))

        # sgdOptimizer = SGD(lr=0.01)
        adamOptimizer = keras.optimizers.Adam(lr=0.1)

        # binaryCrossEntropy = keras.losses.binary_crossentropy(y_true, y_pred)


        model.compile(optimizer=adamOptimizer, loss='mean_squared_error', metrics=['accuracy'])

        
        plot_model(model, to_file='model.png')
    else:
        raise Exception("Model is already available")

    return model



def normalize(d):
    min = numpy.amin(d)
    max = numpy.amax(d)
    dNorm = (d - min)/(max - min) * 2 - 1
    return dNorm

def normalizeData(data):
    # print("shape normalizeData", data.shape)

    global i
    normValues = []
    for d in data:
        dNorm = normalize(d)
        normValues.append(dNorm)
        # i = i + 1
        # plotSpectrogram("name" + str(i), numpy.array(dNorm), True, 'norm')


    return numpy.array(normValues)

def trainModel(xTrain, yTrain):
    global model
    xt = numpy.array(xTrain)
    yt = numpy.array(yTrain)

    # for y in yt:
    #     printOutput(y)
    # print("xt before nor", xt, numpy.max(xt))

    # xt = normalize(xt, axis=-1, order=2)
    # yt = normalize(yt, axis=-1, order=2)

    xtNorm = normalizeData(xt)

    # print("xtNorm", xtNorm, numpy.max(xtNorm), xtNorm.shape)
    # print("yt", yt)
    # print('xTrain', xTrain.shape, xTrain[0].shape, numpy.reshape(-1, ))
    # print('yTrain', yTrain.shape)

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    model.fit(xtNorm, yt, validation_data=(xtNorm, yt), epochs=5, callbacks=[tensorboard_callback])

    print("output at 0", model.layers[0].output)

def predict(xPredict):
    xp = normalizeData(xPredict)
    yPredictions = model.predict(xp)
    # print("max min yPredictions", numpy.max(yPredictions), numpy.min(yPredictions), yPredictions)
    yPredictions = normalizeData(yPredictions)
    # print("max min yPredictions afer norm", numpy.max(yPredictions), numpy.min(yPredictions))

    yApproximation = []
    for y in yPredictions:
        printOutput(y)
        yApproximationItem = [1 if yVal > .9 else 0 for yVal in y]
        yApproximation.append(yApproximationItem)

    # print("yApproximation", yApproximation)

    return yApproximation

