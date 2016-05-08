import math
import random
from node import Node
import time

NUM_CLASS_CODES = 10
BIASED_NODE = 1  # on/off
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
                nodes_per_layer = num_inputs + BIASED_NODE

            # if output layer
            elif layer_index == output_layer_index:
                nodes_per_layer = NUM_CLASS_CODES

            # hidden layer
            else:
                nodes_per_layer = int(num_inputs / 3) + BIASED_NODE  # + 1 for bias node

            # generate small random weights for nodes, then add to layer
            layer_nodes = []
            for node_index in range(nodes_per_layer):  # includes bias node
                node = Node()
                layer_nodes.append(node)

            if BIASED_NODE == 1 and layer_index != output_layer_index:  # if use biased nodes
                layer_nodes[nodes_per_layer - 1].value = 1

            layers.append(layer_nodes)

        # generate weights
        for layer in layers[1:]:
            prev_layer = layers[layers.index(layer) - 1]
            num_weights = len(layer)
            for node in prev_layer:
                node.weights = [(random.random() / 10) + .001 for weight in range(num_weights)]

        self.layers = layers


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_inverse(x):
    return x * (1 - x)


def get_error_gradient(value, weighted_sum):
    return value * (1 - value) * weighted_sum


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


def back_prop_learning(input_vectors, network):
    num_inputs = len(input_vectors[0]) - 1
    network.generate_layers(num_inputs)

    iteration = 0
    num_incorrect = 0
    total_checks = 0

    while iteration < 10:

        for matrix in input_vectors:
            class_code = int(matrix[64])

            # set input node value to this element of the input matrix
            for element_index in range((len(matrix) - 1)):
                network.layers[0][element_index].value = int(matrix[element_index])

            # set expected outputs for the given inputs
            output_layer = network.layers[len(network.layers) - 1]
            for index in range(NUM_CLASS_CODES):
                if index == class_code:
                    output_layer[index].expected = 1
                else:
                    output_layer[index].expected = 0

            # feed forward
            correct_pattern = False
            for layer in network.layers[1:]:
                layer_index = network.layers.index(layer)
                output_layer_index = network.num_hidden_layers + 1

                for node in layer:
                    # inputs are the previous layer's nodes
                    layers = network.layers
                    prev_layer = layers[layers.index(layer) - 1]
                    j_index = layer.index(node)
                    summation = 0
                    for input_node in prev_layer:
                        summation += (input_node.value * input_node.weights[j_index])

                    node.weighted_sum = summation
                    node.value = sigmoid(summation)

            # backpropagate
            for layer in reversed(network.layers[:output_layer_index]):
                prev_layer_index = network.layers.index(layer) + 1
                prev_layer = network.layers[prev_layer_index]

                deltas = []
                for node_j in layer:
                    weighted_sum = 0
                    for node_k in prev_layer:
                        k = prev_layer.index(node_k)

                        if prev_layer_index == output_layer_index:
                            node_k.delta = (node_k.expected - node_k.value) * sigmoid_inverse(node_k.value)

                        weighted_sum += node_j.weights[k] * node_k.delta

                    node_j.delta = get_error_gradient(node_j.value, weighted_sum)

                # update the weights for nodes in layer
                for node_j in layer:
                    for node_k in prev_layer:
                        k = prev_layer.index(node_k)
                        node_j.weights[k] += LEARNING_RATE * node_j.value * node_k.delta

            actual = []
            expected = []

            for node in output_layer:
                expected.append(node.expected)
                actual.append(node.value)
                high_index = actual.index(max(actual))
                if high_index == class_code:
                    correct_pattern = True

            total_checks += 1
            if not correct_pattern:
                num_incorrect += 1

        print("\n\nCorrect Rate: {}".format((total_checks - num_incorrect) / total_checks))
        iteration += 1
        print(iteration)


if __name__ == '__main__':
    training_set = read_file('../resources/optdigits_train.txt')
    # testing_set = read_file('../resources/optdigits_test.txt')
    net = NeuralNet(1)
    back_prop_learning(training_set, net)
