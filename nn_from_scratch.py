import numpy as np
import matplotlib.pyplot as plt
import copy

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
        parameters["W" + str(l)] = np.random.randn(layers[l], layers[l-1]) * np.sqrt(2 / layers[l-1])  #Convergence very slow for normal initialization. Used He initialization
        parameters["b" + str(l)] = np.zeros((layers[l], 1))
    
    return parameters

def forward_linear(A, W, b):
    """
    Input -- A (activations of the previous layer), W (Weights of current layer), b (Bias of current layer)
    Output -- Z (Linear activation of current layer), cache (A, W, b)

    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache

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

#Start of backward propagation
def backward_linear(dZ, linear_cache):
    """
    Input -- dZ, linear_cache
    output -- dW, db, dA_prev

    """
    A_prev, W, b = linear_cache
    m = A_prev.shape[1] #Shape of any A is (nl, m)
    dW = (1 / m) * np.dot(dZ, A_prev.T) ##Assuming that the cost derivative dZ or dA are not scaled by 1/m
    db = (1 / m) * np.sum(dZ, axis = 1, keepdims=True) #Sum of dZ over all examples, since for one example db = dZ
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

#Calculate the derivatives of common functions
#dZ = dA * g'(Z)

#Sigmoid
def sigmoid_backwards(dA, activation_cache):
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    der_sigmoid =  s * (1-s) #Derivative of a sigmoid
    return np.multiply(dA, der_sigmoid)

#Tanh
def tanh_backwards(dA, activation_cache):
    Z = activation_cache
    t = np.tanh(Z)
    der_tanh = 1 - t ** 2
    return np.multiply(dA, der_tanh)

#ReLU
def relu_backwards(dA, activation_cache):
    der_relu = np.where(activation_cache > 0, 1, 0)
    return np.multiply(dA, der_relu)

#Incorporate the derivative of the activation function 
def backward_activation(dA, cache, activation):
    """
    Input -- dA, cache (linear + activation of that layer), activation type of that layer
    output -- dA_prev, dW, db

    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backwards(dA, activation_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)
    
    elif activation == "sigmoid":
        dZ = sigmoid_backwards(dA, activation_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)
    
    elif activation == "tanh":
        dZ = tanh_backwards(dA, activation_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)

    return dA_prev, dW, db

def backward_propagation(AL, Y, caches):
    """
    Input -- AL, Y, caches of all layers
    output -- gradients dictionary containing dA_prev, dW, db

    """

    grads = {}
    L = len(caches) #Number of caches is number of layers other than input
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    #Calculate dAL for first layer from last based on cost function (Ignoring 1/m here, added later)
    dAL = - np.divide(Y, AL) + np.divide((1-Y), (1-AL))

    current_cache = caches[L-1] #Last/Current cache
    dA_prev_temp, dW_temp, db_temp = backward_activation(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp #Activation of previous, i.e L-1 layer. dAL is already known
    grads["dW" + str(L)] = dW_temp #Weights of current Lth layer
    grads["db" + str(L)] = db_temp #Biases of current Lth layer

    #Go through each layer from last but one to first
    for l in range(L-2, -1, -1): #l goes from L-2 to 0
        current_cache = caches[l] 
        dA_prev_temp, dW_temp, db_temp = backward_activation(grads["dA" + str(l+1)], current_cache, "relu") #Takes activation of l+1 layer as input. Eg, L-2 takes L-1 layer as input
        grads["dA" + str(l)] = dA_prev_temp #Activation of previous, i.e 
        grads["dW" + str(l+1)] = dW_temp #Weights of current Lth layer
        grads["db" + str(l+1)] = db_temp #Biases of current Lth layer

        # print("Gradients",  grads["dA" + str(l)],  grads["dW" + str(l+1)], grads["db" + str(l+1)], sep= ", ")

    return grads

#Finally, we update the parameters

def update_parameters(params, grads, learning_rate):

    parameters = copy.deepcopy(params)
    L = len(parameters) // 2

    for l in range(L): #Goes from 0 to L-1, do l+ 1
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (grads["dW" + str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * (grads["db" + str(l+1)])

    return parameters







