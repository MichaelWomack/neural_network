
class Node(object):
    """ class to represent a node, its value, and its weight """

    def __init__(self):
        self.value = None
        self.weights = []
        self.weighted_input_sum = None
        self.delta = None

    def __str__(self):
        return 'Value: {}\t Weight: {}'.format(self.value, self.weights)
