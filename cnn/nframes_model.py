from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.utils import normalize, plot_model
from datetime import datetime
from cnn import printOutput, plotMultipleImages
import tensorflow as tf
import keras
import numpy
import os.path

def normalize(d):
    min = numpy.amin(d)
    max = numpy.amax(d)
    # print("min max", min, max)
    dNorm = (d - min)/(1 if (max - min) == 0 else (max - min)) * 2 - 1
    # print(dNorm, d-min)
    return dNorm
    # return numpy.array([numpy.array([row[0]]) for row in dNorm])

def normalizeData(data):
    # print("shape normalizeData", data.shape)

    normValues = []
    for d in data:
        dNorm = normalize(d)
        normValues.append(dNorm)

    return numpy.array(normValues)

class NFramesModel:

    def __init__(self, charPair = 'ai', noOfEpoch = 20):
        self.charPair = charPair
        self.checkpointPath = "checkpoint/nframes/" + self.charPair + "/"
        self.checkpointFileName = self.checkpointPath + "model.ckpt"
        self.Ylogits = None
        self.sess = None
        self.noOfEpoch = noOfEpoch
        self.createModel()

    def createModel(self):

        if(self.Ylogits == None):
        
            with tf.name_scope('inputAndOutput') as scope:
                self.X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4, 128, 1], name="X")
                self.Y = tf.compat.v1.placeholder(tf.float32, [None, 1], name="Y")
                x = self.X
                
            with tf.name_scope('conv1') as scope:
                # kernelInitializer = keras.initializers.Constant(value=1)
                self.conv1 = tf.compat.v1.keras.layers.Conv2D(
                    filters=1, 
                    kernel_size=(2, 2),
                    strides=(1, 1),
                    activation=None, 
                    data_format='channels_last',
                    # kernel_initializer=kernelInitializer,
                    kernel_initializer='random_uniform',
                )(x)
                self.pool1 = tf.compat.v1.keras.layers.AveragePooling2D(
                    pool_size=(2,2),
                    strides=(2,1),
                    padding="same")(self.conv1)
                # self.conv1Flattening = tf.compat.v1.layers.Flatten(name="conv1Flattening")(self.pool1)
                x = self.pool1

                print("conv1 shape", self.conv1.shape)
                print("pool1 shape", self.pool1.shape)
                # print("conv1Flattening shape", self.conv1Flattening.shape)
                
            with tf.name_scope('conv2') as scope:
                kernelInitializer = keras.initializers.Constant(value=1)
                self.conv2 = tf.compat.v1.keras.layers.Conv2D(
                    filters=1, 
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    activation=None, 
                    data_format='channels_last',
                    # kernel_initializer=kernelInitializer
                    kernel_initializer='random_uniform'
                )(x)
                self.pool2 = tf.compat.v1.keras.layers.AveragePooling2D(
                    pool_size=(2,2),
                    strides=(2,1),
                    padding="valid")(self.conv2)
                x = self.pool2
                
                print("conv2 shape", self.conv2.shape)
                print("pool2 shape", self.pool2.shape)
                
                
            with tf.name_scope('flattening') as scope:
                self.flattenedLayer = tf.compat.v1.layers.Flatten(name="flattening")(x)
                x = self.flattenedLayer

            with tf.name_scope('fullyConnectedLayerOne') as scope:
                print("x.shape", x.shape)
                self.fc1InitialWeights = tf.random.truncated_normal([126, 50], stddev=1e-1, name="fc1InitialWeights")
                self.fc1Weight = tf.Variable(self.fc1InitialWeights, name="fc1Weights")

                self.fc1InitialBias = tf.constant(0.0, shape=[50], name="fc1InitialBias")
                self.fc1Bias = tf.Variable(self.fc1InitialBias, trainable=True, name="fc1Bias")

                matMulAddBias = tf.nn.bias_add(tf.matmul(x, self.fc1Weight), self.fc1Bias)
                self.fc1 = matMulAddBias
                x = self.fc1
                self.fc1 = tf.math.sigmoid(matMulAddBias, name="fc1")
                # self.fc1 = tf.compat.v1.keras.layers.Dropout(0.1, noise_shape=None, seed=None)(self.fc1)

                self.Ylogits = self.fc1

            with tf.name_scope('fullyConnectedLayerTwo') as scope:
                print("x.shape", x.shape)
                self.fc2InitialWeights = tf.random.truncated_normal([50, 1], name="fc2InitialWeights")
                self.fc2Weight = tf.Variable(self.fc2InitialWeights, name="fc2Weights")

                self.fc2InitialBias = tf.constant(0.0, shape=[1], name="fc2InitialBias")
                self.fc2Bias = tf.Variable(self.fc2InitialBias, trainable=True, name="fc2Bias")

                matMulAddBias = tf.nn.bias_add(tf.matmul(self.fc1, self.fc2Weight), self.fc2Bias)
                self.fc2 = matMulAddBias
                self.fc2 = tf.math.sigmoid(matMulAddBias, name="fc2")
                # self.fc2 = tf.compat.v1.keras.layers.Dropout(0.1, noise_shape=None, seed=None)(self.fc2)
                self.Ylogits = self.fc2

            # with tf.name_scope('fullyConnectedLayerThree') as scope:
            #     print("x.shape", x.shape)
            #     self.fc3InitialWeights = tf.random.truncated_normal([100, 1], name="fc3InitialWeights")
            #     self.fc3Weight = tf.Variable(self.fc3InitialWeights, name="fc3Weights")

            #     self.fc3InitialBias = tf.constant(0.0, shape=[1], name="fc3InitialBias")
            #     self.fc3Bias = tf.Variable(self.fc3InitialBias, trainable=True, name="fc3Bias")

            #     matMulAddBias = tf.nn.bias_add(tf.matmul(self.fc2, self.fc3Weight), self.fc3Bias)
            #     self.fc3 = tf.math.sigmoid(matMulAddBias, name="fc3")
            #     self.Ylogits = self.fc3

            with tf.name_scope('crossEntropy'):
                self.crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Ylogits, labels=self.Y)
                # self.crossEntropy = tf.keras.losses.BinaryCrossentropy()(self.Y, self.Ylogits)
                self.lossFunction = tf.reduce_mean(self.crossEntropy)
                # self.lossFunction = tf.keras.losses.MSE(self.Y, self.Ylogits)

            with tf.name_scope('accuracy'):
                self.correctPrediction = tf.equal(tf.math.round(self.Ylogits), self.Y)
                self.accuracy = tf.reduce_mean(tf.cast(self.correctPrediction, tf.float32))

            with tf.name_scope('train'):
                self.trainStep = tf.compat.v1.train.AdamOptimizer(learning_rate=.001).minimize(self.lossFunction)
            
    def normalizeTensor(self, inputTensor):
        # minTensor = tf.reduce_min(inputTensor, axis=1)
        # maxTensor = tf.reduce_max(inputTensor, axis=1)

        return  tf.map_fn(
            lambda x: tf.subtract(
                tf.multiply(
                    tf.div(
                        tf.subtract(
                            x,
                            tf.reduce_min(x)
                        ),
                        tf.subtract(
                            tf.reduce_max(x),
                            tf.reduce_min(x)
                        )
                    ),
                    tf.constant(2.0, shape=[1])
                ),
                tf.constant(1.0, shape=[1])
            ),
            inputTensor
        )

    def trainOrRestore(self, xTrain, yTrain, forceTrain = False):
        if(forceTrain or not self.restoreModel()):
            self.train(xTrain, yTrain)
    
    def train(self, xTrain, yTrain):
        # print("yTrain", yTrain)


        self.sess = tf.compat.v1.Session()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)



        self.createSummaries()

        accHist = []
        for i in range(0, self.noOfEpoch):
            indices = tf.range(start=0, limit=len(xTrain), dtype=tf.int32)
            # print("indices", indices)
            shuffledIndices = tf.random.shuffle(indices)
            # print("shuffledIndices", self.sess.run(shuffledIndices))

            xShuffled = []
            yShuffled = []
            # for si in self.sess.run(shuffledIndices):
            for si in self.sess.run(indices):
                xShuffled.append(xTrain[si])
                yShuffled.append(yTrain[si])

            xNorm = normalizeData(xShuffled)
            xNorm = numpy.reshape(xNorm, (-1, 4, 128, 1))
            # print("xNormReshaped shape", xNorm.shape)
            yShuffled = numpy.reshape(yShuffled, (-1, 1))
            # print("yShuffled shape bfore", yShuffled.shape, yShuffled[0])

            # yShuffled = yShuffled[:,[8]]
            # yShuffled = yShuffled[:,[0, 59, 8]]
            print("yShuffled shape", yShuffled.shape)
            self.sess.run(self.trainStep, feed_dict={self.X: xNorm, self.Y: yShuffled})
            xoutput = self.sess.run(self.X, feed_dict={self.X: xNorm, self.Y: yShuffled})
            # print("X outpu", xoutput, xoutput.shape )


            # normalizedLayer = self.sess.run(
            #     self.normalizedLayer, 
            #     feed_dict={self.X: xNorm, self.Y: yShuffled})
            # print("normalizedLayer outpu", normalizedLayer, normalizedLayer.shape)
            # flattenedLayer = self.sess.run(
            #     self.flattenedLayer, 
            #     feed_dict={self.X: xNorm, self.Y: yShuffled})
            # print("flattenedLayer outpu", flattenedLayer, flattenedLayer.shape)
            # reduceMin = self.sess.run(
            #     tf.reduce_min(self.flattenedLayer), 
            #     feed_dict={self.X: xNorm, self.Y: yShuffled})
            # print("reduceMin outpu", reduceMin, reduceMin.shape)
            # reduceMax = self.sess.run(
            #     tf.reduce_max(self.flattenedLayer), 
            #     feed_dict={self.X: xNorm, self.Y: yShuffled})
            # print("reduceMax outpu", reduceMax, reduceMax.shape)
            fc1 = self.sess.run(
                self.fc1, 
                feed_dict={self.X: xNorm, self.Y: yShuffled})
            # print("fc1", fc1, fc1.shape)
            Ylogits = self.sess.run(
                self.Ylogits, 
                feed_dict={self.X: xNorm, self.Y: yShuffled})
            YlogitsRound = self.sess.run(
                tf.math.round(self.Ylogits), 
                feed_dict={self.X: xNorm, self.Y: yShuffled})
            Y = self.sess.run(
                self.Y, 
                feed_dict={self.X: xNorm, self.Y: yShuffled})
            equal = self.sess.run(
                tf.equal(tf.math.round(self.Ylogits), self.Y),
                feed_dict={self.X: xNorm, self.Y: yShuffled})
            # reduceMean = self.sess.run(
            #     tf.reduce_mean(tf.cast(tf.equal(self.Ylogits, self.Y), tf.float32)),
            #     feed_dict={self.X: xNorm, self.Y: yShuffled})
            # print("equal outpu", Ylogits, Y, YlogitsRound, equal,  Ylogits.shape)
            # lossFunction = self.sess.run(
            #     self.lossFunction, 
            #     feed_dict={self.X: xNorm, self.Y: yShuffled})
            # print("lossFunction outpu", lossFunction, lossFunction.shape)
            # crossEntropy = self.sess.run(
            #     self.crossEntropy, 
            #     feed_dict={self.X: xNorm, self.Y: yShuffled})
            # print("crossEntropy outpu", crossEntropy, crossEntropy.shape)
            # correctPrediction = self.sess.run(
            #     self.correctPrediction, 
            #     feed_dict={self.X: xNorm, self.Y: yShuffled})
            # print("correctPrediction outpu", correctPrediction, correctPrediction.shape)

            acc = self.sess.run([self.accuracy], feed_dict={self.X: xNorm, self.Y: yShuffled})
            accHist.append(acc)

            meanAcc = numpy.mean(accHist)
            print('Epoch number {} Training Accuracy: {}'.format(i+1, meanAcc))

            self.saveImages(self.X, self.charPair + "X", xNorm, yShuffled)
            self.saveImages(self.pool1, self.charPair + "pool1", xNorm, yShuffled)
            # self.saveImages(self.pool1, "pool1", xNorm, yShuffled)

            self.addSummary(xNorm, yShuffled, i)

            if(meanAcc > .78 and i >= 5):
                break
            
        self.saveModel()

    def saveImages(self, layer, layerName, x, y):
        data = self.sess.run(
                    layer, 
                    feed_dict={self.X: x, self.Y: y}
                )
        plotMultipleImages(layerName, data)

    def saveModel(self):
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(self.sess, self.checkpointFileName)
        print("Model saved in path: " + save_path)

    def restoreModel(self):
        if os.path.exists(self.checkpointPath):
            saver = tf.compat.v1.train.Saver()
            self.sess = tf.compat.v1.Session()
            saver.restore(self.sess, self.checkpointFileName)
            print("Model restored from path: " + self.checkpointPath)
            return True
        else:
            print("Model not found in path: " + self.checkpointPath)
            return False


    def predict(self, xPredict):
        xp = normalizeData(xPredict)
        xp = numpy.reshape(xp, (-1, 151, 128, 1))

        self.createSummaries(False)

        yPredictions = self.sess.run(self.Ylogits, feed_dict={self.X: xp})
        # yPredictions = self.sess.run(tf.math.round(self.Ylogits), feed_dict={self.X: xp})

        yApproximation = []
        for y in yPredictions:
            printOutput(y)
            # yApproximationItem = [1 if yVal > .9 else 0 for yVal in y]
            yApproximationItem = [yVal for yVal in y]
            yApproximation.append(yApproximationItem)

        # print("yApproximation", yApproximation)

        return yApproximation

    def createSummaries(self, train = True):
        if(self.sess != None):
            if train:
                self.testTrainLogWriter = tf.compat.v1.summary.FileWriter("logs/train")
            else:
                self.testTrainLogWriter = tf.compat.v1.summary.FileWriter("logs/test")
            self.testTrainLogWriter.add_graph(self.sess.graph)
            

            # tf.compat.v1.summary.image("Input", tf.transpose(self.X, perm=[0, 2, 1, 3]))
            tf.compat.v1.summary.image("Input", self.X, max_outputs=11)
            # tf.compat.v1.summary.image("Conv1", self.getImageFromConvAt(self.conv1, 0), max_outputs=11)
            # tf.compat.v1.summary.image("Pool1", self.pool1, max_outputs=11)
            # tf.compat.v1.summary.image("Conv1", self.conv2, max_outputs=11)
            # tf.compat.v1.summary.image("Pool1", self.pool2, max_outputs=11)

            # tf.compat.v1.summary.scalar('Loss', self.lossFunction)
            # tf.compat.v1.summary.scalar('cross_entropy', self.crossEntropy)
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
        self.testTrainLogWriter.add_summary(summary, epoch)
        self.testTrainLogWriter.flush()
        