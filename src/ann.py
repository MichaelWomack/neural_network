import math
import random
from node import Node

NUM_CLASS_CODES = 10


class NeuralNet(object):
    def __init__(self, num_layers):
        self.num_hidden_layers = num_layers
        self.nodes_per_hl = 0
        self.layers = None

    def generate_layers(self, num_inputs):
        layers = []
        # total layers = num hidden layers + input layer + output layer
        for layer_index in range(self.num_hidden_layers + 2):

            # if input layer
            if layer_index == 0:
                nodes_per_layer = num_inputs

            # if output layer
            elif layer_index == self.num_hidden_layers + 1:
                nodes_per_layer = NUM_CLASS_CODES

            # hidden layer
            else:
                nodes_per_layer = int(num_inputs / 2)

            # generate small random weights for nodes, then add to layer
            layer_nodes = []
            for node_index in range(nodes_per_layer):
                node = Node()
                node.weight = (random.random() / 100) + .001
                layer_nodes.append(node)

            layers.append(layer_nodes)

        self.layers = layers


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_inverse(x):
    return x * (1 + math.exp(-x))


def read_file(file_path):
    input_matrices = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.rstrip('\n')
            matrix = line.split(',')
            input_matrices.append(matrix)

    return input_matrices


def back_prop_learning(input_vectors, network):
    deltas = []

    num_inputs = len(input_vectors[0]) - 1
    network.generate_layers(num_inputs)

    for matrix in input_vectors:
        class_code = matrix[64]

        # set input node value to this element of the input matrix
        # Weight already initialized
        for element_index in range(len(matrix) - 1):
            network.layers[0][element_index].value = matrix[element_index]

        for layer in network.layers[1:]:

            for node in layer:
                # inputs are the previous layer's nodes
                inputs = network.layers[network.index(layer) - 1]
                summation = 0
                for input in inputs:
                    summation += (input.value * input.weight)

                node.value = sigmoid(summation)


if __name__ == '__main__':
    inputs = read_file('../resources/optdigits_train.txt')
    net = NeuralNet(2)
    back_prop_learning(inputs, net)
