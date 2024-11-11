import numpy as np
import time

class DeepNeuralNetwork():
    def __init__(self, sizes, activation='sigmoid', dropout_rate=0.5):
        self.sizes = sizes
        self.dropout_rate = dropout_rate
        # Choose activation function
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise ValueError("Activation function is currently not support, please use 'relu' or 'sigmoid' instead.")
        
        self.params = self.initialize()
        # Save all intermediate values, i.e. activations
        self.cache = {}
        
    def relu(self, x, derivative=False):
        '''
            Derivative of ReLU is a bit more complicated since it is not differentiable at x = 0
        
            Forward path:
            relu(x) = max(0, x)
            In other word,
            relu(x) = 0, if x < 0
                    = x, if x >= 0

            Backward path:
            ∇relu(x) = 0, if x < 0
                     = 1, if x >=0
        '''
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        '''
            Forward path:
            σ(x) = 1 / 1+exp(-z)
            
            Backward path:
            ∇σ(x) = exp(-z) / (1+exp(-z))^2
        '''
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def initialize(self):
        input_layer=self.sizes[0]
        hidden_layer=self.sizes[1]
        output_layer=self.sizes[2]
        
        params = {
            "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1./input_layer),
            "b1": np.zeros((hidden_layer, 1)) * np.sqrt(1./input_layer),
            "W2": np.random.randn(output_layer, hidden_layer) * np.sqrt(1./hidden_layer),
            "b2": np.zeros((output_layer, 1)) * np.sqrt(1./hidden_layer)
        }
        return params
    
    def feed_forward(self, x, training):
        self.cache["X"] = x
        self.cache["Z1"] = np.dot(self.params["W1"], self.cache["X"].T) + self.params["b1"]
        self.cache["A1"] = self.activation(self.cache["Z1"])

        if training:
            self.cache["dropout_mask"] = (np.random.rand(*self.cache["A1"].shape) > self.dropout_rate).astype(float)
            self.cache["A1"] *= self.cache["dropout_mask"]
        else:
            self.cache["A1"] *= (1 - self.dropout_rate)  # Scale at test time

        self.cache["Z2"] = np.dot(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        self.cache["A2"] = self.softmax(self.cache["Z2"])
        return self.cache["A2"]
    
    def back_propagate(self, y, output, l_rate):
        current_batch_size = y.shape[0]

        dZ2 = output - y.T
        dW2 = (1./current_batch_size) * np.dot(dZ2, self.cache["A1"].T)
        db2 = (1./current_batch_size) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.params["W2"].T, dZ2)
        
        dZ1 = dA1 * self.activation(self.cache["Z1"], derivative=True)
        dZ1 *= self.cache["dropout_mask"]

        dW1 = (1./current_batch_size) * np.dot(dZ1, self.cache["X"])
        db1 = (1./current_batch_size) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {"W1": self.params["W1"] - dW1 * l_rate, "b1": self.params["b1"] - db1 * l_rate, "W2": self.params["W2"] - dW2 * l_rate, "b2": self.params["b2"] - db2 * l_rate}
        return self.grads
    
    def cross_entropy_loss(self, y, output):
        l_sum = np.sum(np.multiply(y.T, np.log(output)))
        m = y.shape[0]
        l = -(1./m) * l_sum
        return l
                
    def accuracy(self, y, output):
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))

    def train(self, x_train, y_train, x_test, y_test, epochs=100, 
              batch_size=128, l_rate=0.1):
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // self.batch_size)
        
        start_time = time.time()
        template = "Epoch {}: {:.3f}s, train acc={:.3f}, train loss={:.3f}, test acc={:.3f}, test loss={:.3f}"
        
        # Train
        for i in range(self.epochs):
            # Shuffle
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for j in range(num_batches):
                # Batch
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]
                
                # Forward
                output = self.feed_forward(x, True)
                # Backprop
                self.params = self.back_propagate(y, output, l_rate)

            # Evaluate performance
            # Training data
            if i % 10 == 0:
                output = self.feed_forward(x_train, False)
                train_acc = self.accuracy(y_train, output)
                train_loss = self.cross_entropy_loss(y_train, output)
                # Test data
                output = self.feed_forward(x_test, False)
                test_acc = self.accuracy(y_test, output)
                test_loss = self.cross_entropy_loss(y_test, output)
                print(template.format(i+1, time.time()-start_time, train_acc, train_loss, test_acc, test_loss))
        output = self.feed_forward(x_train, False)
        train_acc = self.accuracy(y_train, output)
        train_loss = self.cross_entropy_loss(y_train, output)
        output = self.feed_forward(x_test, False)
        test_acc = self.accuracy(y_test, output)
        test_loss = self.cross_entropy_loss(y_test, output)
        print(template.format(self.epochs, time.time()-start_time, train_acc, train_loss, test_acc, test_loss))