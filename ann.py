__author__ = 'sondremare'

import theano
import theano.tensor as T
import theano.tensor.nnet as Tann
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor.nnet as Tann


def createRandomWeights(size):
    return theano.shared(np.random.uniform(-.1, .1, size=size))


class ann:
    def __init__(self, hiddenLayers, lr=.1):
        self.inputLayerSize = 3  # 786
        self.outputLayerSize = 3  # 10
        self.lrate = lr
        layers = hiddenLayers
        layers.insert(0, self.inputLayerSize)
        layers.append(self.outputLayerSize)
        #self.build_ann(layers)
        self.build_simple_ann()

    def build_simple_ann(self):
        input = T.dvector('input')
        w1 = createRandomWeights((786, 600))
        w2 = createRandomWeights((600, 600))
        w3 = createRandomWeights((600, 10))
        b1 = createRandomWeights((600))
        b2 = createRandomWeights((600))
        b3 = createRandomWeights((10))
        x1 = Tann.sigmoid(T.dot(input, w1) + b1)
        x2 = Tann.sigmoid(T.dot(x1, w2) + b2)
        x3 = Tann.sigmoid(T.dot(x2, w3) + b3)
        error = T.sum((input - x3) ** 2)
        params = [w1, b1, w2, b2, w3, b3]
        gradients = T.grad(error, params)
        backprop_acts = [(p, p - self.lrate * g) for p, g in zip(params, gradients)]
        self.predictor = theano.function([input], [x3, x2, x1])
        self.trainer = theano.function([input], error, updates=backprop_acts)

    def build_ann(self, layers):
        input = T.dvector('input')
        weights = []
        biases = []
        activation_values = [T.dvector('input')]
        for i, val in enumerate(layers):
            if i != 0:
                weights.append(createRandomWeights(layers[i - 1], layers[i]))
                biases.append(createRandomWeights(layers[i]))
                activation_value = Tann.sigmoid(T.dot(activation_values[-1], weights[-1]) + biases[1])
                activation_values.append(activation_value)

        error = T.sum((activation_values[0] - activation_values[-1]) ** 2)
        params = []
        for i, val in enumerate(weights):
            params.append(weights[i])
            params.append(biases[i])

        gradients = T.grad(error, params)
        backprop_acts = [(p, p - self.lrate * g) for p, g in zip(params, gradients)]
        self.predictor = theano.function([input], [x2, x1])
        self.trainer = theano.function([input], error, updates=backprop_acts)


