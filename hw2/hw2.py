import numpy as np
from torchvision.datasets import MNIST
from sklearn.preprocessing import OneHotEncoder

np.set_printoptions(threshold=np.inf)

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

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

train_X = train_X / 255.0
test_X = test_X / 255.0

def one_hot_encode(labels, num_classes=10):
    one_hot_encoded = np.zeros((labels.shape[0], num_classes))
    one_hot_encoded[1, 2] = 3
    one_hot_encoded[np.arange(labels.shape[0]), labels] = 1
    return one_hot_encoded

train_Y_one_hot = one_hot_encode(train_Y)
test_Y_one_hot = one_hot_encode(test_Y)

np.random.seed(42)  

input_size = 784  
output_size = 10  

W = np.random.randn(input_size, output_size) * 0.01 
b = np.zeros(output_size)

print (W.shape)
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(X, W, b):
    z = np.dot(X, W) + b
    return softmax(z)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss

def backward(X, y_true, y_pred, W, b, learning_rate):
    m = y_true.shape[0]
    
    dz = y_pred - y_true
    
    # Gradients
    dW = np.dot(X.T, dz) / m
    db = np.sum(dz, axis=0) / m
    
    # Update weights and biases
    W -= learning_rate * dW
    b -= learning_rate * db
    
    return W, b

def train(train_X, train_Y, W, b, epochs, batch_size, learning_rate):
    for epoch in range(epochs):
        permutation = np.random.permutation(train_X.shape[0])
        train_X_shuffled = train_X[permutation]
        train_Y_shuffled = train_Y[permutation]
        
        for i in range(0, train_X.shape[0], batch_size):
            X_batch = train_X_shuffled[i:i+batch_size]
            Y_batch = train_Y_shuffled[i:i+batch_size]
            
            y_pred = forward(X_batch, W, b)
            
            loss = cross_entropy_loss(y_pred, Y_batch)
            
            W, b = backward(X_batch, Y_batch, y_pred, W, b, learning_rate)
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return W, b

def accuracy(X, Y, W, b):
    y_pred = forward(X, W, b)
    predictions = np.argmax(y_pred, axis=1)
    labels = np.argmax(Y, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy

initial_accuracy = accuracy(test_X, test_Y_one_hot, W, b)
print(f"Initial accuracy: {initial_accuracy * 100:.2f}%")

W, b = train(train_X, train_Y_one_hot, W, b, epochs=100, batch_size=100, learning_rate=0.01)

final_accuracy = accuracy(test_X, test_Y_one_hot, W, b)
print(f"Final accuracy: {final_accuracy * 100:.2f}%")
