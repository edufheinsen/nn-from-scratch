import network


class FullyConnectedNet:
    def __init__(self):
        self.W1, self.b1 = network.he_initialization(784, 16)
        self.W2, self.b2 = network.xavier_initialization(16, 10)

    def forward(self, x):
        x = network.relu(self.W1 @ x + self.b1)
        x = network.log_softmax(self.W2 @ x + self.b2)
        return x

    def backward(self):
        pass

    def update(self):
        pass
