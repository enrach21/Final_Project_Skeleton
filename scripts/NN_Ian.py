import numpy as np


class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        self.hidden_neurons = 3
        # initialize weights as .50 for simplicity
        # self.weights = np.array([[.50], [.50], [.50]])
        self.weights1 = np.random.rand(self.inputs.shape[1], self.hidden_neurons) # input to hidden layer
        self.weights2 = np.random.rand(self.hidden_neurons, self.outputs.shape[1]) # hidden layer to output
        self.error_history = []
        self.epoch_list = []
        
    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
    
    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights1))
        self.predicted = self.sigmoid(np.dot(self.hidden, self.weights2))
        
    # going backwards through the network to update weights
    def backpropagation(self):
        self.error  = self.outputs - self.predicted
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.hidden.T, (2*(self.outputs - self.predicted) * self.sigmoid(self.predicted, True)))
        d_weights1 = np.dot(self.inputs.T,  (np.dot(2*(self.outputs - self.predicted) * self.sigmoid(self.predicted, True), self.weights2.T) * self.sigmoid(self.hidden, True)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        
    # train the neural net for 1000 iterations
    def train(self, epochs=10000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()    
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)
   
            
    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights1))
        self.predicted = self.sigmoid(np.dot(self.hidden, self.weights2))
        return self.predicted
    
    
class NeuralNetwork1:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        self.hidden_neurons = 3
        # initialize weights as .50 for simplicity
        # self.weights = np.array([[.50], [.50], [.50]])
        self.weights1 = np.random.rand(self.inputs.shape[0], self.hidden_neurons) # input to hidden layer
        self.weights2 = np.random.rand(self.hidden_neurons, 1) # hidden layer to output
        self.error_history = []
        self.epoch_list = []
        
    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
    
    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights1))
        self.predicted = self.sigmoid(np.dot(self.hidden, self.weights2))
        
    # going backwards through the network to update weights
    def backpropagation(self):
        self.error  = self.outputs - self.predicted
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.hidden.T, (2*(self.outputs - self.predicted[0]) * self.sigmoid(self.predicted[0], True)))
        d_weights1 = np.dot(self.inputs.T,  (np.dot(2*(self.outputs - self.predicted[0]) * self.sigmoid(self.predicted[0], True), self.weights2.T) * self.sigmoid(self.hidden, True)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        
    # train the neural net for 1000 iterations
    def train(self, epochs=10000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()    
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)
   
            
    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights1))
        self.predicted = self.sigmoid(np.dot(self.hidden, self.weights2))
        return self.predicted
    
    
class NeuralNetwork2:
    
    def __init__(self, inputs, outputs):
        self.LR = 2
        self.inputs  = inputs
        self.outputs = outputs
        self.hidden_neurons = 3
        self.m = self.outputs.shape[1]
        # initialize weights as .50 for simplicity
        # self.weights = np.array([[.50], [.50], [.50]])
        self.weights1 = np.random.rand( self.hidden_neurons, self.inputs.shape[0]) # input to hidden layer
        self.weights2 = np.random.rand(self.outputs.shape[0] ,self.hidden_neurons) # hidden layer to output
        self.error_history = []
        self.epoch_list = []
        
        #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
     # data will flow through the neural network.
    def feed_forward(self):
        Z1 = np.dot(self.weights1, self.inputs)
        self.A1 = sigmoid(Z1)
        Z2 = np.dot(self.weights2, self.A1)
        self.A2 = sigmoid(Z2)
        
    # going backwards through the network to update weights
    def backpropagation(self):
        self.error  = self.outputs - self.A2
        # logprobs = np.mulitply(np.log(self.predicted), self.outputs) + np.multiply((1-self.outputs),np.log(1-self.predicted))
        # cost = -np.sum(logprobs)/self.m
        
        dZ2 = self.A2 - self.outputs
        dW2 = (1/self.m) * np.dot(dZ2, self.A1.T)
        dZ1 = np.multiply(np.dot(self.weights2.T,dZ2), 1 - np.power(self.A1, 2))
        dW1 = (1 / self.m) * np.dot(dZ1, self.inputs.T)
        
        self.weights1 = self.weights1 - self.LR * dW1
        self.weights2 = self.weights2 - self.LR * dW2
        
    def train(self, epochs=10000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()    
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)
   
            
    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        self.A1 = self.sigmoid(np.dot(self.inputs, self.weights1))
        self.A2 = self.sigmoid(np.dot(self.A1, self.weights2))
        return self.S2
    
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

class NeuralNetwork3:
    
    def __init__(self, inputs, outputs, activation='sigmoid'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
            
        self.LR = 2
        self.inputs  = inputs
        self.outputs = outputs
        self.hidden_neurons = 3
        # initialize weights as .50 for simplicity
        # self.weights = np.array([[.50], [.50], [.50]])
        self.W1 = np.random.rand( self.inputs.shape[1], self.hidden_neurons) # input to hidden layer
        self.W2 = np.random.rand(self.hidden_neurons ,self.outputs.shape[1]) # hidden layer to output
        self.error_history = []
        self.epoch_list = []
        
    def forward(self):
        #forward propagation through our network
        self.z1 = np.dot(self.inputs, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.A1 = self.activation(self.z1) # activation function
        self.z2 = np.dot(self.A1, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        self.A2 = self.activation(self.z2) # final activation function
            
    def backward(self):
        self.error = self.outputs - self.A2
        dW2 = self.error * self.activation_prime (self.A2)
        A1_error= dW2.dot(self.W2.T)
        dW1=A1_error*self.activation_prime(self.A1)
            
        self.W1 += self.inputs.T.dot(dW1)
        self.W2 += self.A1.T.dot(dW2)
        
    def train(self, epochs=10000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.forward()
            # go back though the network to make corrections based on the output
            self.backward()    
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)
    
    def predict(self, new_input):
        self.A1 = self.activation(np.dot(self.inputs, self.W1))
        self.A2 = self.activation(np.dot(self.A1, self.W2))
        return self.A2
            
    
    
class NeuralNetwork4:
    
    def __init__(self, inputs, outputs, activation='sigmoid'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
            
        self.LR = 2
        self.inputs  = inputs
        self.outputs = outputs
        self.hidden_neurons = 3
        self.W1 = np.random.rand( self.inputs.shape[1], self.hidden_neurons) # input to hidden layer
        self.W2 = np.random.rand(self.hidden_neurons ,1) # hidden layer to output
        self.error_history = []
        self.epoch_list = []
        
    def forward(self):
        #forward propagation through our network
        self.z1 = np.dot(self.inputs, self.W1) # dot product of X (input) and first set of
        self.A1 = self.activation(self.z1) # activation function
        self.z2 = np.dot(self.A1, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        self.A2 = self.activation(self.z2) # final activation function
        # print('A1')
        # print(self.A1.shape)
        # print('A2')
        # print(self.A2.shape)
            
    def backward(self):
        # print('output')
        # print(self.outputs.shape)
        # print('A2')
        # print(self.A2.shape)
        self.error = self.outputs - self.A2
        # print(self.error.shape)
        dW2 = self.error * self.activation_prime (self.A2)
        # print('dW2')
        # print(dW2.shape)
        A1_error= dW2.dot(self.W2.T)
        # print('A1_error')
        # print(A1_error.shape)
        dW1=A1_error*self.activation_prime(self.A1)
        # print('dW1')
        # print(dW1.shape)  

        self.W1 += self.inputs.T.dot(dW1)
        self.W2 += self.A1.T.dot(dW2)
        # self.W1 = self.W1 - self.LR * dW1
        # self.W2 = self.W2 - self.LR * dW2
        
    def train(self, epochs=10000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.forward()
            # go back though the network to make corrections based on the output
            self.backward()    
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)
    
    def predict(self, new_input):
        self.A1 = self.activation(np.dot(new_input.T, self.W1))
        print('test')
        print(self.A1)
        self.A2 = self.activation(np.dot(self.A1, self.W2))
        return self.A2