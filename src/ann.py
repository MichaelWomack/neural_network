import math
import random
from node import Node

NUM_CLASS_CODES = 10
BIASED_NODE = 1 # per layer
LEARNING_RATE = .1

class NeuralNet(object):
    def __init__(self, hidden_layers):
        self.num_hidden_layers = hidden_layers
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
                nodes_per_layer = int(num_inputs / 2) + BIASED_NODE # + 1 for bias node

            # generate small random weights for nodes, then add to layer
            layer_nodes = []
            for node_index in range(nodes_per_layer): # includes bias node
                node = Node()
                layer_nodes.append(node)

            layers.append(layer_nodes)

        # generate weights
        for layer in layers[1:]:
            prev_layer = layers[layers.index(layer) - 1]
            num_weights = len(prev_layer) * len(layer)
            for node in prev_layer:
                node.weights = [(random.random() / 100) + .001 for weight in range(num_weights)]

        self.layers = layers


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

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
    return weighted_sum


def back_prop_learning(input_vectors, network, backprop=True):
    deltas = []

    num_inputs = len(input_vectors[0]) - 1
    network.generate_layers(num_inputs)

    for matrix in input_vectors[:1]:
        class_code = matrix[64]

        # set input node value to this element of the input matrix
        # Weight already initialized
        for element_index in range(len(matrix) - 1):
            network.layers[0][element_index].value = int(matrix[element_index])


        for layer in network.layers[1:]:
            print("Layer: {}\tLength: {}".format(network.layers.index(layer), len(layer)))
            for node in layer:
                # inputs are the previous layer's nodes
                layers = network.layers
                inputs = network.layers[layers.index(layer) - 1]
                j_index = layer.index(node)
                summation = 0
                for input in inputs:
                    i_index = inputs.index(input)
                    summation += (input.value * input.weights[i_index * j_index])

                node.weighted_sum = summation
                node.value = sigmoid(summation)
                print(node.value)


        # ### back propagation ###
        # # for each node in output layer
        #     # delta_k = err_k * inverse(in_k)
        #     #update rule --> weight from

        # get error gradient for output nodes
        output_layer_index = len(network.layers) - 1
        output_layer = network.layers[output_layer_index]
        for node in output_layer:
            # get node error gradient (output layer)
            delta = (int(class_code) - node.value) * sigmoid_inverse(node.value)
            node.delta = delta


        # from output layer to input

        # hidden layer -- output layer gradient == learning rate * current node value
        for layer in reversed(network.layers[:output_layer_index]):
            prev_layer_index = network.layers.index(layer) + 1
            prev_layer = network.layers[prev_layer_index]

            for node_j in layer:

                for node_k in prev_layer:

                    j = layer.index(node_j)
                    k = prev_layer.index(node_k)

                    node_j.delta = get_error_gradient(node_j, node_k)
                    node_j.weights[j * k] += LEARNING_RATE * node_j.value * node_k.delta
                    print(node_j.delta)
                    print(node_j.weights)










if __name__ == '__main__':
    inputs = read_file('../resources/optdigits_train.txt')
    net = NeuralNet(1)
    back_prop_learning(inputs, net)
