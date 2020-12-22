import numpy as np

import network


class FullyConnectedNet:
    """
    Fully connected neural network architecture.

    Architecture is FC -> RELU -> FC -> LOG_SOFTMAX, with NLL loss

    Attributes:
        W1: (16, 784) array containing the weights in the first fully connected layer
        b1: (16, 1) array containing the biases in the first fully connected layer
        W2: (10, 16) array containing the weights in the second fully connected layer
        b2: (10, 1) array containing the biases in the second fully connected layer
        parameters: list of all learnable parameters (weights and biases) in the network
        X: (784, batch_size) array of input data
        A1: (16, batch_size) array of first FC layer ReLU activations
        A2: (10, batch_size) array of second FC layer LogSoftmax activations
        target: (batch_size, ) array containing the correct classes for each sample in the minibatch
        L: negative log-likelihood loss averaged across all samples in the batch being processed
        dW1: (16, 784) array containing the first FC layer weights' gradients
        db1: (16, 1) array containing the first FC layer biases' gradients
        dW2: (10, 16) array containing the second FC layer weights' gradients
        db2: (10, 1) array containing the second FC layer biases' gradients
        gradients: list of all learnable parameter gradients
        learning_rate: the learning rate used for minibatch stochastic gradient descent
    """
    def __init__(self, learning_rate: int = 0.01):
        """
        Initializes network parameters, gradients, hyperparameters, and cached values for backprop
        """
        self.W1, self.b1 = network.he_initialization(784, 16)
        self.W2, self.b2 = network.xavier_initialization(16, 10)
        self.parameters = [self.W1, self.b1, self.W2, self.b2]
        self.X, self.A1, self.A2 = None, None, None
        self.target = None
        self.L = None
        self.dW1, self.db1, self.dW2, self.db2 = None, None, None, None
        self.gradients = None
        self.learning_rate = learning_rate

    def forward(self, X: np.ndarray, target: np.ndarray):
        """
        Carries out one forward pass
        """
        self.X = X
        Z1 = self.W1 @ self.X + self.b1
        self.A1 = network.relu(Z1)
        Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = network.log_softmax(Z2)
        self.L = network.nll_loss(self.A2, target)
        self.target = target
        return self.A2, self.L

    def backward(self):
        """
        Carries out one backward pass
        """
        m = self.A2.shape[1]
        dA2 = np.zeros(self.A2.shape)
        dA2[self.target.astype(int), np.arange(m).astype(int)] = 1
        dZ2 = np.exp(self.A2) - dA2
        self.db2 = np.mean(dZ2, axis=1)[:, None]
        self.dW2 = dZ2 @ self.A1.T
        dA1 = self.W2.T @ dZ2
        relu_grad = np.ones(self.A1.shape)
        relu_grad[self.A1 == 0] = 0
        dZ1 = dA1 * relu_grad
        self.db1 = np.mean(dZ1, axis=1)[:, None]
        self.dW1 = dZ1 @ self.X.T
        self.gradients = [self.dW1, self.db1, self.dW2, self.db2]

    def update_parameters(self):
        """
        Updates all learnable parameters using stochastic gradient descent
        """
        self.parameters = [self.parameters[i] - self.learning_rate * self.gradients[i]
                           for i in range(len(self.parameters))]
        self.W1, self.b1, self.W2, self.b2 = self.parameters
