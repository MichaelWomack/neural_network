
class NeuralNetwork:
    """
        Class to create an artificial neural network
    """

    def __init__(self):
        self.input_matrices = []
        self.input_classes = []
        self.inputs = []
        self.hidden_layers = []

    def read_file(self):
        file_lines = open('../resources/optdigits_train.txt', 'r').readlines()
        for line in file_lines:
            line = line.rstrip('\n')
            matrix = line.split(',')
            print("{}\n{}".format(len(matrix), matrix))
            self.input_matrices.append(matrix)


if __name__ == '__main__':
    net = NeuralNetwork()
    net.read_file()