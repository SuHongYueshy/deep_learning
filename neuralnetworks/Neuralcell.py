"""神经元"""
import numpy as np


class NeuralNode(object):
    def _init_(self,activation, input_size, output_size = 1):
        self.activation = activation
        self._weights_shape = (input_size, output_size)
        self.weights_ = []

    def cale(self, input):
        self._weights_ = np.random.rand(self._weights_shape)
        sum = np.sum(np.dot(input, self._weights_))
        return self._activation_(sum)

class NeuralLayer(object):
    def def_init_(self, input_size, neural_node_count, activation,output_size):
        self._neuralnodes_ = [NeuralNode(activation, input_size, output_size) for _ in range(neural_node_count)]

    def calc(self, input_data):
        return map(lambda node: node.calc(input_data), self._neuralnodes_)
