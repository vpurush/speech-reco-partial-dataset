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

def normalize(d):
    min = numpy.amin(d)
    max = numpy.amax(d)
    dNorm = (d - min)/(max - min) * 2 - 1
    return dNorm

def normalizeData(data):
    # print("shape normalizeData", data.shape)

    normValues = []
    for d in data:
        dNorm = normalize(d)
        normValues.append(dNorm)

    return numpy.array(normValues)

class FunctionalModel:

    def __init__(self):
        self.Ylogits = None
        self.createModel()

    def createModel(self):

        if(self.Ylogits == None):
        
            with tf.name_scope('inputAndOutput') as scope:
                self.X = tf.compat.v1.placeholder(tf.float32, shape=(None, 151, 128), name="X")
                self.Y = tf.compat.v1.placeholder(tf.float32, [None, 26*26], name="Y")
                
            with tf.name_scope('flattening') as scope:
                self.flattenedLayer = tf.compat.v1.layers.Flatten(name="flattening")(self.X)
                x = self.flattenedLayer

            with tf.name_scope('fullyConnectedLayerOne') as scope:
                print("x.shape", x.shape)
                self.fc1InitialWeights = tf.random.truncated_normal([151 * 128, 26 * 26], name="fc1InitialWeights")
                self.fc1Weight = tf.Variable(self.fc1InitialWeights, name="fc1Weights")

                self.fc1InitialBias = tf.constant(1.0, shape=[26 * 26], name="fc1InitialBias")
                self.fc1Bias = tf.Variable(self.fc1InitialBias, trainable=True, name="fc1Bias")

                matMulAddBias = tf.nn.bias_add(tf.matmul(x, self.fc1Weight), self.fc1Bias)
                self.fc1 = tf.math.sigmoid(matMulAddBias, name="fc1")
                self.Ylogits = self.fc1

            with tf.name_scope('crossEntropy'):
                self.crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Ylogits, labels=self.Y)
                self.lossFunction = tf.reduce_mean(self.crossEntropy)

            with tf.name_scope('accuracy'):
                self.correctPrediction = tf.equal(tf.argmax(self.Ylogits, 0), tf.argmax(self.Y, 0))
                self.accuracy = tf.reduce_mean(tf.cast(self.correctPrediction, tf.float32))

            with tf.name_scope('train'):
                self.trainStep = tf.compat.v1.train.AdamOptimizer(learning_rate=10).minimize(self.lossFunction)
            
    
    def train(self, xTrain, yTrain):
        print("yTrain", yTrain)

        noOfEpoch = 50

        self.sess = tf.compat.v1.Session()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        accHist = []
        for i in range(0, noOfEpoch):
            self.sess.run(self.trainStep, feed_dict={self.X: normalizeData(xTrain), self.Y: yTrain})

            acc = self.sess.run([self.accuracy], feed_dict={self.X: normalizeData(xTrain), self.Y: yTrain})
            accHist.append(acc)

            print('Epoch number {} Training Accuracy: {}'.format(i+1, numpy.mean(accHist)))


    def predict(self, xPredict):
        xp = normalizeData(xPredict)
        yPredictions = self.sess.run(self.Ylogits, feed_dict={self.X: normalizeData(xp)})

        yApproximation = []
        for y in yPredictions:
            printOutput(y)
            yApproximationItem = [1 if yVal > .9 else 0 for yVal in y]
            yApproximation.append(yApproximationItem)

        # print("yApproximation", yApproximation)

        return yApproximation