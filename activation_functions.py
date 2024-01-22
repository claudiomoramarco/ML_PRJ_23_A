import numpy as np
from scipy.special import expit

"""
Let's define the activation functions that we'll use:
linear, ReLU, sigmoid, tanh and softmax, and their derivatives
that we'll use in the backpropagation
"""

def linear(x):
    return x

def linear1(x):
    return 1

def relu(input):
    input = np.ravel(input)
    toRet = []
    for x in input:
        toRet.append(np.maximum(0,x))
    return toRet

def relu1(input):
    input = np.ravel(input)
    toRet = []
    for x in input:
        if x <= 0:
            toRet.append(0)
        else:
            toRet.append(1)
    return toRet



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid1(input):
    sig = sigmoid(input)
    sig = np.array(sig)
    return sig * (1-sig)

def tanh(x):
    return np.tanh(x)

def tanh1(x):
    return 1-pow(np.tanh,2)
    
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


# Non so se è giusto
def softmax1(x):
    s = softmax(x)
    return s * (np.eye(len(s)) - s.reshape(-1, 1))



def derivative(function):
    if function==linear:
        return linear1
    if function==relu:
        return relu1
    if function==sigmoid:
        return sigmoid1
    if function==tanh:
        return tanh1
    if function==softmax:
        return softmax1
    else:
        print("Derivative error")
        exit()