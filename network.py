import numpy as np
from typing import Tuple


def he_initialization(input_features: int, output_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    He initialization of weight matrix and bias vector (uses Gaussian distribution for weights).
    :param input_features
    :param output_features
    :return: weight matrix of shape (output_features, input_features), bias vector of shape
    (output_features, 1)
    """
    weights = np.random.randn(output_features, input_features) * np.sqrt(2 / input_features)
    biases = np.zeros((output_features, 1))
    return weights, biases


def xavier_initialization(input_features: int, output_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Xavier initialization of weight matrix and bias vector (uses Uniform distribution for weights).
    :param input_features
    :param output_features
    :return: weight matrix of shape (output_features, input_features), bias vector of shape
    (output_features, 1)
    """
    weights = np.random.uniform(-np.sqrt(1 / input_features), np.sqrt(1 / input_features),
                                size=(output_features, input_features))
    biases = np.zeros((output_features, 1))
    return weights, biases


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation function.
    """
    return np.maximum(x, 0)


def log_softmax(x: np.ndarray) -> np.ndarray:
    """
    LogSoftmax activation function.
    """
    a = np.max(x)
    logsumexp = a + np.log(np.sum(np.exp(x - a)))
    return x - logsumexp


def negative_log_likelihood_loss(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Negative log-likelihood loss function.
    :param prediction: output of log-softmax layer
    :param target: 1-D array of correct classes for each example
    :return: NLL loss
    """
    nll_loss = np.mean(prediction[target, np.arange(prediction.shape[1])])
    return nll_loss
