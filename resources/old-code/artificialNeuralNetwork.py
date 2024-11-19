#https://www.youtube.com/watch?v=0oWnheK-gGk&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=6&ab_channel=ValerioVelardo-TheSoundofAI

import numpy as np
from random import random

# save the activations and derivatives (for back propagation)
# implement backpropagation
# implement gradient descent
# implement train 
# train our net with some dummy dataset
# make some predictions

class MLP: #Class Multilayer Perception

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2): # constructor al que le pasamos el número de entradas, de capas ocultas (lita integers con el número de neuronas en cada capa) y salidas
        """
        Constructor for the MLP. Takes the number of inputs, 
            a variable number of hidden layers, and number of outputs

        Args:
            num_inputs (int) : Number of inputs
            hidden_layers (list) : A list of ints for the hidden layers
            num_outputs (int) : Number of outputs
        """
       
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # Create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # initiate random connection weights for the layers
        weights = []

        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1]) # matriz de pesos aleatoria de tamaño numeroinputs x numero de neuronas en la capa
            weights.append(w)

        self.weights = weights

        # save activations per layer
        activations = []

        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a) # will be a list of arrays where each array on the list represents the activations for a given layer

        self.activations = activations

        # save derivatives per layer
        derivatives = []

        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)

        self.derivatives = derivatives






    def forward_propagate(self, inputs): # function for compute propagation between layers
        """Computes forward propagation of the network based on input signals.

        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropagation
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            # calculate net inputs
            net_inputs = np.dot(activations, w)

            # calculate the activations of the next step
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations

        return activations
    


    def back_propagate(self, error, verbose=False): # function for send error back propagated towards the layers on the left
        """Backpropogates an error signal.
        Args:
            error (ndarray): The error to backprop.
        Returns:
            error (ndarray): The final error of the input
        """
        # Matemáticamente:
        # dE/dW_i = (y - a_[i+1]) s'(h_[i+1]) a_i
        # s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1]

        # dE/dW_[i-1] = ((y - a_[i+1]) s'(h_[i+1])) W_i s'(h_i) a_[i-1]

        for i in reversed(range(len(self.derivatives))): # recorre el rango de fin a inicio -> de izda a dcha en las capas
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations) # ndarray ([[0.1], [0.2]]) -->> ndarrya([0.1, 0.2]) 
            current_activations = self.activations[i] # ndarrya([0.1, 0.2]) -->> ndarray ([[0.1], [0.2]])

            # Rearrange delta so we get a matrix vector of size (Xcol, 1row)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T # .T --->>> Transposed matrix
            # Rearrange current_activations so we get a matrix vector of size (1col, Xrow)
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)

            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

    
    def _sigmoid(self, x):
        return 1/(1+ np.exp(-x))


    def _sigmoid_derivative(self, x):
        return x * (1.0-x)  # s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))    


    def gradient_descent(self, learning_rate):
        
        for i in range(len(self.weights)):
            weights = self.weights[i]
            #print("Original W{} {}".format(i, weights))
            derivatives = self.derivatives[i]

            weights += derivatives * learning_rate
            #print("Updated W{} {}".format(i, weights))

    
    def train(self, inputs, targets, epochs, learning_rate): # inputs and targets are our data set; epochs
        
        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):

                # forward propagate
                output = self.forward_propagate(input)

                # calculate error
                error = target - output

                # back propagate
                self.back_propagate(error)

                # apply gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            # report error for each epoch
            print("Error: {} at epoch {}".format(sum_error / len(inputs), i))


    def _mse(self, target, output): # mean sqared error -> the average of the sqared error
        return np.average((target - output)**2)
        

if __name__ ==  "__main__":
    # create an MLP
    mlp = MLP(2, [5], 1) # se pueden cambiar aqui los valores que le pasamos a la función

    # create some inputs
    #inputs = np.random.rand(mlp.num_inputs)

    # perform forward propagation
    #outputs = mlp.forward_propagate(inputs)

    # print results
    #print("The network input is: {}".format(inputs))
    #print("The network output is: {}".format(outputs))
    

    # create dataset to train a network for the sum operation
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])
    
    # train our mlp
    mlp.train(inputs, targets, 50, 0.1)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    output = mlp.forward_propagate(input)
    print()
    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))