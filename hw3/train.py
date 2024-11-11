import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from model import DeepNeuralNetwork
from torchvision.datasets import MNIST


# Settings
parser = argparse.ArgumentParser(description='Neural Networks from Scratch')
parser.add_argument('--activation', action='store', dest='activation', required=False, default='sigmoid', help='activation function: sigmoid/relu')
parser.add_argument('--batch_size', action='store', dest='batch_size', required=False, default=64)
parser.add_argument('--l_rate', action='store', dest='l_rate', required=False, default=1e-3, help='learning rate')
parser.add_argument('--beta', action='store', dest='beta', required=False, default=.9, help='beta in momentum optimizer')
args = parser.parse_args()

# Helper function
def show_images(image, num_row=2, num_col=5):
    # plot images
    image_size = int(np.sqrt(image.shape[-1]))
    image = np.reshape(image, (image.shape[0], image_size, image_size))
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(image[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
def one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)

def main():
    # Load data
    print("Loading data...")
    train_X, y = download_mnist(True)
    test_X, test_Y = download_mnist(False)
    # Normalize
    print("Preprocessing data...")
    x_train = train_X / 255.0
    x_test = test_X / 255.0
    # One-hot encode labels
    num_labels = 10

    y_train = one_hot(y.astype('int32'), num_labels)
    y_test = one_hot(test_Y.astype('int32'), num_labels)

    # Split, reshape, shuffle
    train_size = 60000
    shuffle_index = np.random.permutation(train_size)
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
    print("Training data: {} {}".format(x_train.shape, y_train.shape))
    print("Test data: {} {}".format(x_test.shape, y_test.shape))
#     show_images(x_train)

    # Train
    print("Start training!")
    print (args)
    dnn = DeepNeuralNetwork(sizes=[784, 100, 10], activation=args.activation)
    dnn.train(x_train, y_train, x_test, y_test)
    
if __name__ == '__main__':
    main()