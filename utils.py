import gzip
import numpy as np
import matplotlib.pyplot as plt
import itertools


def load_mnist():
    """
    Read in MNIST dataset from "data" directory.

    Return:
        X_train - (60000, 784) NumPy array containing the training images
        X_test - (10000, 784) NumPy array containing the testing images
        y_train - (60000, ) NumPy array containing the training data labels
        y_test - (10000, ) NumPy array containing the testing data labels
    """

    for x, y in list(itertools.product(["train", "test"], ["images", "labels"])):
        image_size = 28
        if x == "train":
            num_images = 60000
            images_path = "data/train-images-idx3-ubyte.gz"
            labels_path = "data/train-labels-idx1-ubyte.gz"
        else:
            num_images = 10000
            images_path = "data/t10k-images-idx3-ubyte.gz"
            labels_path = "data/t10k-labels-idx1-ubyte.gz"
        if y == "images":
            f = gzip.open(images_path, "r")
            f.read(16)
            buf = f.read(image_size * image_size * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(num_images, image_size * image_size)
            if x == "train":
                X_train = data
            else:
                X_test = data
        else:
            f = gzip.open(labels_path, 'r')
            f.read(8)
            buf = f.read(num_images)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            if x == "train":
                y_train = labels
            else:
                y_test = labels

    return X_train, X_test, y_train, y_test


def make_batches(data, labels, batch_size):
    """
    Create batches from shuffled data.

    Args:
        data - (n_samples, n_features) NumPy array containing image data without labels
        labels - (n_samples, ) NumPy array containing image labels
        batch_size - batch size (int)

    Returns:
        batches - list of tuples where each tuple contains a (batch_size, n_features) Numpy
                  array containing image data and a (batch_size, ) NumPy array containing the
                  corresponding labels
    """
    labels = labels[:, np.newaxis]
    concatenated = np.hstack((data, labels))
    np.random.shuffle(concatenated)
    new_data = concatenated[:, :-1]
    new_labels = concatenated[:, -1]
    breaks = [batch_size * i for i in range(1, (concatenated.shape[0] - 1) // batch_size + 1)]
    split_data = np.array_split(new_data, breaks)
    split_labels = np.array_split(new_labels, breaks)
    batches = list(zip(split_data, split_labels))
    return batches


def display_mnist_image(image):
    """
    Displays image of digit in the MNIST dataset

    Args:
        image - (784, ) NumPy array containing MNIST image
    """
    plt.imshow(image.reshape(28, 28))
    plt.show()
