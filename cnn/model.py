from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense
import numpy

model = None
def createModel():
    global model
    if(model == None):
        model = Sequential()
        model.add(Conv1D(filters=5, kernel_size=4, strides=1, activation='relu', input_shape=(149, 40)))
        model.add(Flatten())
        model.add(Dense(units=26*26, activation='relu'))

        sgdOptimizer = SGD(lr=0.01)
        model.compile(optimizer=sgdOptimizer, loss='mean_squared_error', metrics=['accuracy'])
    else:
        raise Exception("Model is already available")

    return model

def trainModel(xTrain, yTrain):
    xt = numpy.array(xTrain)
    yt = numpy.array(yTrain)
    # print('xTrain', xTrain.shape, xTrain[0].shape, numpy.reshape(-1, ))
    # print('yTrain', yTrain.shape)
    model.fit(xt, yt, validation_data=(xt, yt), epochs=30)