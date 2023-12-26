import numpy as np
from abc import ABC, abstractmethod

"""
Let's define the activation functions that we'll use:
linear, ReLU, sigmoid, tanh and softmax, and their derivatives
that we'll use in the backpropagation
"""

def linear(x):
    return x

def linear1(x):
    return 1

def relu(x):
    return np.maximum(0,x)

def relu1(x):
    if x <= 0:
        return 0
    else: 
        return 1
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid1(x):
    return sigmoid(x) * (1 - sigmoid(x))

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

"""
Definiamo ora le classi per le funzioni di attivazione
La classe Activation definisce il modello di base,
che dividiamo nei metodi activation e derivative.
La classe Activation non deve
essere istanziata direttamente ma viene usata solo
per definire le sottoclassi che ci interessano.
"""

class Activation(ABC):

    @abstractmethod
    def activation(self, x):
        """
        Calcola l'output della funzione di attivazione.
        """
        pass

    @abstractmethod
    def derivative(self, x):
        """
        Calcola la derivata della funzione di attivazione.
        """
        pass

class LinearActivation(Activation):
    def activation(self, x):
        return linear(x)

    def derivative(self, x):
        return linear1(x)

class ReLUActivation(Activation):
    def activation(self, x):
        return relu(x)

    def derivative(self, x):
        return relu1(x)

class SigmoidActivation(Activation):
    def activation(self, x):
        return sigmoid(x)

    def derivative(self, x):
        return sigmoid1(x)

class TanhActivation(Activation):
    def activation(self, x):
        return tanh(x)
    
    def derivative(self, x):
        return tanh1(x)

class SoftmaxActivation(Activation):
    def activation(self, x):
        return softmax(x)

    def derivative(self, x):
        return softmax1(x)