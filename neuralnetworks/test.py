import numpy as np
from neuralnetworks import Neuralcell


def sigmoid(x):
    return 1.0/(1+np.exp(-x))

input_data = [[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
layer1 = Neuralcell.NeuralLayer(3, 3, sigmoid, 1)
layer2 = Neuralcell.NeuralLayer(3, 10, sigmoid, 1)
layer3 = Neuralcell.NeuralLayer(10, 1, sigmoid, 1)
r = layer3.calc(layer2.calc(layer1.calc(np.array(input_data))))
for item in r:
    print(item)