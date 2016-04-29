import random
import math


class NeuralNetwork(object):
    """
        Class to represent an artificial neural network
    """

    def __init__(self):
        self.layers = []
        self.weights = []
        self.NUM_HIDDEN_LAYERS = 1
        self.NODES_PER_LAYER = 0

    # activation function
    def sigmoid_func(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_inverse(self, x):
        return x * (1 + math.exp(-x))


# returns data lines in lists from given file name
def read_file(file_path):
    input_matrices = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.rstrip('\n')
            matrix = line.split(',')
            input_matrices.append(matrix)

    return input_matrices


# trains neural network
def back_prop_learning(input_data, network):
    # network inputs each have an input vector x and output vector y

    deltas = []

    network.NODES_PER_LAYER = int(len(input_data) / 2)
# until criterion satisfied do

    # initialize weights
    for i in range(len(input_data)):  # i
        for j in range(network.NODES_PER_LAYER):  # j
            # .001 - .01
            weight = (random.random() / 100) + .001
            network.weights.append(weight)  # appended at index i * j

    for matrix in input_data:

        # Propagate inputs forward to compute outputs
        inputs = []
        for element in matrix[:len(matrix) - 1]:  # NOT THE LAST ELEMENT, class code
            inputs.append(element)

        # for each hidden layer --> 1 for now l for L
        layers = []
        for layer_index in range(1, network.NUM_HIDDEN_LAYERS + 2):  # includes output layer

            # if layer_index is > 0 --> num nodes = NODES PER LAYER
            # if layer_index == L (output layer) nodes are --> 0-9

            num_layer_nodes = network.NODES_PER_LAYER
            if layer_index == network.NUM_HIDDEN_LAYERS + 1:  # output layer has 10 nodes
                num_layer_nodes = 10

            layer_outputs = []
            for node_index in range(num_layer_nodes):  # -->  j elements

                # determine correct number of inputs for this layer
                if layer_index == 1:
                    num_layer_inputs = len(inputs)  # first hidden layer has different number of inputs
                else:
                    num_layer_inputs = network.NODES_PER_LAYER

                summation = 0
                for i in range(num_layer_inputs):  # all the input nodes feeding into this layer
                    summation = summation + (network.weights[i * node_index] * int(inputs[i]))

                output_value = network.sigmoid_func(summation)
                layer_outputs.append(output_value)

            layers.append(layer_outputs) # a sub j values


            # Propagate deltas backward from output layer to input layer

        for node_index in range(10):  # for each output layer node
            class_code = matrix[64]
            # calculate deltas

if __name__ == '__main__':
    ann = NeuralNetwork()
    inputs = read_file('../resources/optdigits_train.txt')

    back_prop_learning(inputs, ann)
