import random, math


class NeuralNetwork(object):
    """
        Class to represent an artificial neural network
    """

    def __init__(self):
        self.layers = []
        self.weights = []
        self.NUM_HIDDEN_LAYERS = 1  # includes output layer
        self.NODES_PER_LAYER = 0


    def init_weights(self):
        pass

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
        for layer_index in range(1, network.NUM_HIDDEN_LAYERS):

            # if layer_index is > 0 --> num nodes = NODES PER LAYER
            # if layer_index == L (output layer) nodes are --> 0-9

            hidden_layer_outputs = []
            for node_index in range(network.NODES_PER_LAYER):  # -->  j elements

                summation = 0
                if layer_index != 0:
                    num_hidden_layer_nodes = network.NODES_PER_LAYER
                else:
                    num_hidden_layer_nodes = len(inputs)

                for i in range(num_hidden_layer_nodes):  # all the input node indices
                    summation = summation + (network.weights[i * node_index] * int(inputs[i]))

                output_value = network.sigmoid_func(summation)
                hidden_layer_outputs.append(output_value)

            # Propagate deltas backward from output layer to input layer




if __name__ == '__main__':
    ann = NeuralNetwork()
    inputs = read_file('../resources/optdigits_train.txt')

    back_prop_learning(inputs, ann)
