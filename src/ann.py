import math
import random
from node import Node
import time

NUM_CLASS_CODES = 10
BIASED_NODE = 1  # per layer
LEARNING_RATE = .1


class NeuralNet(object):
    def __init__(self, hidden_layers):
        self.num_hidden_layers = hidden_layers
        self.nodes_per_hl = 0
        self.layers = None

    def generate_layers(self, num_inputs):
        layers = []
        output_layer_index = self.num_hidden_layers + 1

        # total layers = num hidden layers + input layer + output layer
        for layer_index in range(self.num_hidden_layers + 2):

            # if input layer
            if layer_index == 0:
                nodes_per_layer = num_inputs

            # if output layer
            elif layer_index == output_layer_index:
                nodes_per_layer = NUM_CLASS_CODES

            # hidden layer
            else:
                nodes_per_layer = int(num_inputs / 2) + BIASED_NODE  # + 1 for bias node

            # generate small random weights for nodes, then add to layer
            layer_nodes = []
            for node_index in range(nodes_per_layer):  # includes bias node
                node = Node()
                layer_nodes.append(node)

            layers.append(layer_nodes)

        # generate weights
        for layer in layers[1:]:
            prev_layer = layers[layers.index(layer) - 1]
            num_weights = len(layer)
            for node in prev_layer:
                node.weights = [(random.random() / 1000) + .001 for weight in range(num_weights)]

        self.layers = layers


def sigmoid(x):
    return 1 / float((1 + math.exp(-x)))


def sigmoid_inverse(x):
    activation = sigmoid(x)
    return activation * (1 - activation)


# derivative
def derivative(x):
    return math.exp(x) / pow((math.exp(x) + 1), 2)


def read_file(file_path):
    input_matrices = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.rstrip('\n')
            matrix = line.split(',')
            input_matrices.append(matrix)

    return input_matrices


def get_error_gradient(node_j, node_k):
    weighted_sum = 0
    for weight in node_j.weights:
        weighted_sum += weight * node_k.delta

    return node_j.value * (1 - node_j.value) * weighted_sum



def back_prop_learning(input_vectors, network):
    num_inputs = len(input_vectors[0]) - 1
    network.generate_layers(num_inputs)

    iteration = 0
    while iteration < 3:

        num_incorrect = 0
        total_checks = 0

        for matrix in input_vectors:
            class_code = int(matrix[64])

            # set input node value to this element of the input matrix
            # Weight already initialized
            for element_index in range(len(matrix) - 1):
                network.layers[0][element_index].value = int(matrix[element_index])

            # set expected outputs for the given inputs
            output_layer = network.layers[len(network.layers) - 1]
            for index in range(NUM_CLASS_CODES):
                output_layer[index].expected = 0
                if index == (class_code - 1):
                    output_layer[class_code - 1].expected = 1

            # feed forward
            for layer in network.layers[1:]:
                # print("Layer: {}\tLength: {}".format(network.layers.index(layer), len(layer)))
                for node in layer:
                    # inputs are the previous layer's nodes
                    layers = network.layers
                    inputs = network.layers[layers.index(layer) - 1]
                    j_index = layer.index(node)
                    summation = 0
                    for input in inputs:
                        summation += (input.value * input.weights[j_index])

                    node.weighted_sum = summation
                    node.value = sigmoid(summation)

            # calculate delta, evaluate correct
            output_layer_index = len(network.layers) - 1
            output_layer = network.layers[output_layer_index]
            correct_pattern = True

            for node in output_layer:

                # Sigmoid asymptotic, so never reach 0 or 1.
                if node.value > .9:
                    node.value = 1
                elif node.value < .1:
                    node.value = 0

                if node.expected == node.value:
                    correct_pattern = False

                delta = (node.expected - node.value) * sigmoid_inverse(node.value)
                node.delta = delta

            total_checks += 1
            if not correct_pattern:
                num_incorrect += 1


            # hidden layer -- output layer gradient == learning rate * current node value
            for layer in reversed(network.layers[:output_layer_index]):
                prev_layer_index = network.layers.index(layer) + 1
                prev_layer = network.layers[prev_layer_index]

                deltas = []
                for node_j in layer:

                    for node_k in prev_layer:
                        node_j.delta = get_error_gradient(node_j, node_k)
                        deltas.append(LEARNING_RATE * node_j.value * node_k.delta)

                # update the weights for nodes in layer
                for node_j in layer:
                    j = layer.index(node_j)
                    for index in range(len(prev_layer)):
                        node_j.weights[index] += deltas[index * j]
                #

        print("Correct Rate: {}".format((total_checks - num_incorrect) / total_checks))
        print("Num Incorrect {}: Total: {}".format(num_incorrect, total_checks))
        time.sleep(5)
        iteration += 1


if __name__ == '__main__':
    inputs = read_file('../resources/optdigits_train.txt')
    net = NeuralNet(1)
    back_prop_learning(inputs, net)
