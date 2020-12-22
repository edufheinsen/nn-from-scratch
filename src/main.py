from utils import load_mnist, make_batches, display_mnist_image
from FullyConnectedNet import FullyConnectedNet
import numpy as np
from tqdm import tqdm

batch_size = 32
learning_rate = 0.005
n_epochs = 20

X_train, X_test, y_train, y_test = load_mnist()
X_train = X_train / 255
X_test = X_test / 255
np.random.seed(42)
train_batches = make_batches(X_train, y_train, batch_size)
test_batches = make_batches(X_test, y_test, batch_size)

model = FullyConnectedNet(learning_rate)

for i in tqdm(range(n_epochs)):
    epoch_loss = 0
    for x_batch, y_batch in train_batches:
        pred, loss = model.forward(x_batch.T, target=y_batch)
        model.backward()
        model.update_parameters()
        epoch_loss += loss
    print("training loss for epoch", (i+1), "is", epoch_loss / y_train.size)

total_correct = 0
for x_batch, y_batch in test_batches:
    predicted, loss = model.forward(x_batch.T, target=y_batch)
    predicted_classes = np.argmax(predicted, axis=0)
    actual = y_batch.astype(int)
    correct = np.sum(actual == predicted_classes)
    total_correct += correct

print("test set accuracy is", total_correct / y_test.size)


