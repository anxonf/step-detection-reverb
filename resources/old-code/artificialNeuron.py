#https://www.youtube.com/watch?v=qxIaW-WvLDU&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=3&ab_channel=ValerioVelardo-TheSoundofAI
import math

def sigmoid(x):
    y = 1.0 / (1+ math.exp(-x)) # math exp es la función exponencial; sigmoide = 1/(1+exp^-x)
    return y

def activate(inputs, weights):
    h = 0
    # perform net input
    for x, w, in zip(inputs, weights): # funcion zip iterates the indexes by itself at the same time
        h += x*w # h = sum(xi*wi) = x1*w1 + x2*w2 + x3*w3 -> Producto escalar de los dos vectores x y w --> h= x.w
    # perform actuvation using function sigmoid
    return sigmoid(h)

if __name__ == "__main__":
    inputs = [.5, .3, .2] #inputs and weights are simple lists of the inputs and their associated weigths to go to the neuron
    weights = [.4, .7, .2]
    output = activate(inputs, weights) # Fase de activación
    print(output)
