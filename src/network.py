from typing import Tuple

import numpy as np


def he_initialization(input_features: int, output_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes a weight matrix and a bias vector using He initialization.

    Args:
        input_features: number of features in the input vector
        output_features: number of features in the output vector

    Returns:
        weights: (output_features, input_features) array, with values initialized from
                 N(0, 2 / input_features)
        bias: vector of shape (output_features, 1), with values initialized to 0
    """
    weights = np.random.randn(output_features, input_features) * np.sqrt(2 / input_features)
    bias = np.zeros((output_features, 1))
    return weights, bias


def xavier_initialization(input_features: int, output_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes a weight matrix and a bias vector using Xavier initialization.

    Args:
        input_features: number of features in the input vector
        output_features: number of features in the output vector

    Returns:
        weights: (output_features, input_features) array, with values initialized from
                 U(-1 / sqrt(input_features), 1 / sqrt(input_features))
        bias: vector of shape (output_features, 1), with values initialized to 0
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

    Args:
        x: (num_classes, num_samples) array containing raw (not normalized)
           scores for each class

    Returns:
        (num_classes, num_samples) array containing the logarithms of the normalized
        softmax probabilities
    """
    a = np.max(x, axis=0)  # To avoid potential numerical instability issues
    log_sum_exp = a + np.log(np.sum(np.exp(x - a), axis=0))
    return x - log_sum_exp


def nll_loss(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Negative log-likelihood loss function.

    Args:
        prediction: (num_classes, num_samples) array containing the log probabilities
                    of each class for each sample (i.e. the output of log_softmax())
        target: (num_samples, ) array containing the actual class each sample belongs to

    Returns:
        The average (across all samples) of the negative log probabilities assigned to the correct
        class in each of the samples
    """
    return -np.mean(prediction[target.astype(int), np.arange(prediction.shape[1]).astype(int)])
