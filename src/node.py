
class Node(object):
    """ class to represent a node, its value, and its weight """

    def __init__(self):
        self.value = None
        self.weight = None
        self.weighted_sum = None

    def __str__(self):
        return 'Value: {}\t Weight: {}'.format(self.value, self.weight)
