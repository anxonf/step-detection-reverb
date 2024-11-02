#https://www.youtube.com/watch?v=0oWnheK-gGk&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=6&ab_channel=ValerioVelardo-TheSoundofAI

import numpy as np

class MLP: #Class Multilayer Perception

    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2): # constructor al que le pasamos el número de entradas, de capas ocultas (lita integers con el número de neuronas en cada capa) y salidas
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        #initiate random weights
        self.weights = []

        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1]) # matriz de pesos aleatoria de tamaño numeroinputs x numero de neuronas en la capa
            self.weights.append(w)

    def forward_propagate(self, inputs): # function for compute propagation between layers

        activations = inputs

        for w in self.weights:
            # calculate net inputs
            net_inputs = np.dot(activations, w)

            # calculate the activations of the next step
            activations = self._sigmoid(net_inputs)

        return activations
    
    def _sigmoid(self, x):
        return 1/(1+ np.exp(-x))
    
    
if __name__ ==  "__main__":
    # create an MLP
    mlp = MLP() # se pueden cambiar aqui los valores que le pasamos a la función

    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward propagation
    outputs = mlp.forward_propagate(inputs)

    # print results
    print("The network input is: {}".format(inputs))
    print("The network output is: {}".format(outputs))
    