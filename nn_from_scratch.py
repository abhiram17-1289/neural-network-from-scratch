import numpy as np
import matplotlib.pyplot as plt

#Here I am going to implement a deep neural network from scratch
#This is based on my understandings and learning from the Deep Learning Specialization by Deeplearning.AI
#Also using this code to practice Git

def initialize_params(layers):
    """
    Input -- Layers which is the list of the number of neurons in each Layer of the NN
    Output -- Returns randomly initialized parameters

    """

    parameters = {}
    L = len(layers)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn((layers[l], layers[l-1])) * 0.01
        parameters["b" + str(l)] = np.random.randn((layers[l], 1)) * 0.01
    
    return parameters

def forward_linear(A, W, b):
    """
    Input -- A (activations of the previous layer), W (Weights of current layer), b (Bias of current layer)
    Output -- Z (Linear activation of current layer), cache (A, W, b)

    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)

#Let us define the common activation functions

#Sigmoid
def sigmoid(X):
    return (1 / (1 + np.exp(-X))), X

#Tanh
def tanh(X):
    return np.tanh(X), X

#Relu
def relu(X):
    return np.maximum(0, X), X

def forward_activation(A_prev, W, b, activation):
    """
    Input -- A_prev, W, b, activation
    Output -- A of current layer and cache (linear (A_prev, W, B) and activation (Z))

    """
    
    if activation == "sigmoid":
        Z, linear_cache = forward_linear(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "tanh":
        Z, linear_cache = forward_linear(A_prev, W, b)
        A, activation_cache = tanh(Z)
    
    elif activation == "relu":
        Z, linear_cache = forward_linear(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

#Combine forward_linear and forward_activation for implementing forward_prop for L Layers
def forward_propagation(X, parameters):
    """
    Input -- X, parameters
    Output -- AL(yhat), caches (list of caches of each layer)

    """
    caches = []
    A = X
    L = len(parameters) // 2 #(Two parameters for each layer, this L is not taken from 'layers')

    #Considering L-1 relu layers and Lth layer sigmoid for now. Hopefully will customize this in the future.
    #Loop from 1 to L-1 layers
    for l in range(1, L):
        A_prev = A #Initializes A_prev to the A obtained at the end of this loop. 
        A, cache = forward_activation(A_prev, parameters["W"+ str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache) #Stores the cache for each layer from 1 to L-1
        #Final A will be AL

    #Considering final layer to have sigmoid activation. Hopefully will customize this to take softmax etc in the future
    AL, cache = forward_activation(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache) # Final Cache

    return AL, caches

#Computing Cost function considering sigmoid in the last layer - logistic loss

def compute_cost(AL, Y):
    """
    Input -- AL (predictions), Y (true output values)
    Output -- Cost

    """

    m = Y.shape[1] #Y is of shape (1,m)

    cost = (-1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL))) #Calculates cost of each example by element wise multiplication and then sums them all
    cost = np.squeeze(cost)

    return cost


