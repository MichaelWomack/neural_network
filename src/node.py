
class Node(object):
    """ class to represent a node, its value, and its weight """

    def __init__(self):
        self.value = None
        self.weights = []
        self.weighted_input_sum = None
        self.delta = None
        self.expected = None

    def __str__(self):
        return 'Value: {}\t Weights: {}\tExpected: {}'.format(self.value, self.weights, self.expected)
