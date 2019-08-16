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
    # return numpy.array([numpy.array([row[0]]) for row in dNorm])

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
        self.sess = None
        self.createModel()

    def createModel(self):

        if(self.Ylogits == None):
        
            with tf.name_scope('inputAndOutput') as scope:
                self.X = tf.compat.v1.placeholder(tf.float32, shape=[None, 151, 128, 1], name="X")
                self.Y = tf.compat.v1.placeholder(tf.float32, [None, 26*26], name="Y")
                x = self.X
                
            with tf.name_scope('conv1') as scope:
                kernelInitializer = keras.initializers.Constant(value=0)
                self.conv1 = tf.compat.v1.keras.layers.Conv2D(
                    filters=1, 
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation=None, 
                    data_format='channels_last',
                    kernel_initializer=kernelInitializer)(x)
                self.pool1 = tf.compat.v1.keras.layers.MaxPooling2D(
                    pool_size=(2,2),
                    strides=(2,2),
                    padding="valid")(self.conv1)
                # self.conv1Flattening = tf.compat.v1.layers.Flatten(name="conv1Flattening")(self.pool1)
                x = self.pool1

                print("conv1 shape", self.conv1.shape)
                print("pool1 shape", self.pool1.shape)
                # print("conv1Flattening shape", self.conv1Flattening.shape)
                
            # with tf.name_scope('conv2') as scope:
            #     self.conv2 = tf.compat.v1.keras.layers.Conv1D(filters=5, kernel_size=4, activation=None)(x)
            #     self.pool2 = tf.compat.v1.keras.layers.MaxPooling1D(pool_size=4)(self.conv2)
            #     x = self.pool2
                
            #     print("conv2 shape", self.conv2.shape)
            #     print("pool2 shape", self.pool2.shape)
                
            with tf.name_scope('flattening') as scope:
                self.flattenedLayer = tf.compat.v1.layers.Flatten(name="flattening")(x)
                x = self.flattenedLayer

            with tf.name_scope('fullyConnectedLayerOne') as scope:
                print("x.shape", x.shape)
                self.fc1InitialWeights = tf.random.truncated_normal([4662, 26 * 26], stddev=1e-1, name="fc1InitialWeights")
                self.fc1Weight = tf.Variable(self.fc1InitialWeights, name="fc1Weights")

                self.fc1InitialBias = tf.constant(1.0, shape=[26 * 26], name="fc1InitialBias")
                self.fc1Bias = tf.Variable(self.fc1InitialBias, trainable=True, name="fc1Bias")

                matMulAddBias = tf.nn.bias_add(tf.matmul(x, self.fc1Weight), self.fc1Bias)
                self.fc1 = matMulAddBias
                x = self.fc1
                self.fc1 = tf.math.sigmoid(matMulAddBias, name="fc1")
                self.Ylogits = self.fc1

            # with tf.name_scope('fullyConnectedLayerTwo') as scope:
            #     print("x.shape", x.shape)
            #     self.fc2InitialWeights = tf.random.truncated_normal([26 * 26 * 4, 26 * 26], name="fc2InitialWeights")
            #     self.fc2Weight = tf.Variable(self.fc2InitialWeights, name="fc2Weights")

            #     self.fc2InitialBias = tf.constant(1.0, shape=[26 * 26], name="fc2InitialBias")
            #     self.fc2Bias = tf.Variable(self.fc2InitialBias, trainable=True, name="fc2Bias")

            #     matMulAddBias = tf.nn.bias_add(tf.matmul(x, self.fc2Weight), self.fc2Bias)
            #     self.fc2 = tf.math.sigmoid(matMulAddBias, name="fc2")
            #     self.Ylogits = self.fc2

            with tf.name_scope('crossEntropy'):
                self.crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Ylogits, labels=self.Y)
                self.lossFunction = tf.reduce_mean(self.crossEntropy)

            with tf.name_scope('accuracy'):
                self.correctPrediction = tf.equal(tf.argmax(self.Ylogits, 0), tf.argmax(self.Y, 0))
                self.accuracy = tf.reduce_mean(tf.cast(self.correctPrediction, tf.float32))

            with tf.name_scope('train'):
                self.trainStep = tf.compat.v1.train.AdamOptimizer(learning_rate=.1).minimize(self.lossFunction)
            
    
    def train(self, xTrain, yTrain):
        # print("yTrain", yTrain)

        noOfEpoch = 2

        self.sess = tf.compat.v1.Session()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)



        self.createSummaries()

        accHist = []
        for i in range(0, noOfEpoch):
            indices = tf.range(start=0, limit=len(xTrain), dtype=tf.int32)
            # print("indices", indices)
            shuffledIndices = tf.random.shuffle(indices)
            # print("shuffledIndices", self.sess.run(shuffledIndices))
            # mappedShuffledIndices = numpy.fromiter(map(lambda x: numpy.array([x]), self.sess.run(shuffledIndices)), dtype=object)
            # print("mappedShuffledIndices", mappedShuffledIndices)

            xShuffled = []
            yShuffled = []
            for si in self.sess.run(shuffledIndices):
                xShuffled.append(xTrain[si])
                yShuffled.append(yTrain[si])

            xNorm = normalizeData(xShuffled)
            xNorm = numpy.reshape(xNorm, (-1, 151, 128, 1))
            print("xNormReshaped shape", xNorm.shape)
            yShuffled = numpy.array(yShuffled)
            self.sess.run(self.trainStep, feed_dict={self.X: xNorm, self.Y: yShuffled})
            xoutput = self.sess.run(self.X, feed_dict={self.X: xNorm, self.Y: yShuffled})
            # print("X outpu", xoutput, xoutput.shape )
            conv1output = self.sess.run(self.conv1, feed_dict={self.X: xNorm, self.Y: yShuffled})
            # print("conv1 outpu", conv1output, conv1output.shape)

            acc = self.sess.run([self.accuracy], feed_dict={self.X: xNorm, self.Y: yShuffled})
            accHist.append(acc)

            print('Epoch number {} Training Accuracy: {}'.format(i+1, numpy.mean(accHist)))

            self.addSummary(xNorm, yShuffled, i)


    def predict(self, xPredict):
        xp = normalizeData(xPredict)
        yPredictions = self.sess.run(self.Ylogits, feed_dict={self.X: xp})

        yApproximation = []
        for y in yPredictions:
            # printOutput(y)
            yApproximationItem = [1 if yVal > .9 else 0 for yVal in y]
            yApproximation.append(yApproximationItem)

        # print("yApproximation", yApproximation)

        return yApproximation

    def createSummaries(self):
        if(self.sess != None):
            self.trainLogWriter = tf.compat.v1.summary.FileWriter("logs/train")
            self.testLogWriter = tf.compat.v1.summary.FileWriter("logs/test")
            self.trainLogWriter.add_graph(self.sess.graph)
            

            tf.compat.v1.summary.image("Input", tf.transpose(self.X, perm=[0, 2, 1, 3]))

            tf.compat.v1.summary.scalar('Loss', self.lossFunction)
            tf.compat.v1.summary.scalar('Accuracy', self.accuracy)
            tf.compat.v1.summary.histogram("fc1Weights", self.fc1Weight)
            tf.compat.v1.summary.histogram("fc1Bias", self.fc1Bias)
            # tf.compat.v1.summary.histogram("fc2Weight", self.fc2Weight)
            # tf.compat.v1.summary.histogram("fc2Bias", self.fc2Bias)
            self.logWriter = tf.compat.v1.summary.merge_all()

        else:
            raise Exception("Session not available")

    def addSummary(self, x, y, epoch):

        summary = self.sess.run(self.logWriter, feed_dict={self.X: x, self.Y: y})
        self.trainLogWriter.add_summary(summary, epoch)
        self.trainLogWriter.flush()
        