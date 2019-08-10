from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense
from keras.utils import normalize
import numpy

model = None
def createModel():
    global model
    if(model == None):
        model = Sequential()
        model.add(Conv1D(filters=5, kernel_size=4, strides=1, activation='relu', input_shape=(151, 128)))
        model.add(Flatten())
        model.add(Dense(units=26*26 + 2, activation='sigmoid'))

        sgdOptimizer = SGD(lr=0.01)
        model.compile(optimizer=sgdOptimizer, loss='mean_squared_error', metrics=['accuracy'])
    else:
        raise Exception("Model is already available")

    return model

def normalizeData(data):
    normValues = []
    for d in data:
        dNorm = d - numpy.min(d, axis=0)
        dNorm /= numpy.ptp(dNorm, axis=0)
        normValues.append(dNorm)

    return numpy.array(normValues)

def trainModel(xTrain, yTrain):
    xt = numpy.array(xTrain)
    yt = numpy.array(yTrain)
    print("xt before nor", xt, numpy.max(xt))

    # xt = normalize(xt, axis=-1, order=2)
    # yt = normalize(yt, axis=-1, order=2)

    xtNorm = normalizeData(xt)

    print("xtNorm", xtNorm, numpy.max(xtNorm), xtNorm.shape)
    print("yt", yt)
    # print('xTrain', xTrain.shape, xTrain[0].shape, numpy.reshape(-1, ))
    # print('yTrain', yTrain.shape)
    model.fit(xtNorm, yt, validation_data=(xtNorm, yt), epochs=5)


def predict(xPredict):
    xp = numpy.array(xPredict)
    xp = normalizeData(xp)
    yPredictions = model.predict(xp)
    print("max min yPredictions", numpy.max(yPredictions), numpy.min(yPredictions))
    yPredictions = normalizeData(yPredictions)
    print("max min yPredictions afer norm", numpy.max(yPredictions), numpy.min(yPredictions))

    yApproximation = []
    for y in yPredictions:
        yApproximationItem = [1 if yVal > .7 else 0 for yVal in y]
        yApproximation.append(yApproximationItem)

    # print("yApproximation", yApproximation)

    return yApproximation