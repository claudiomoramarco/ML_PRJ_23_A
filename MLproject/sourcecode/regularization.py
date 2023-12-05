from typing import Any
import numpy as np

"""
Definiamo la classe dei regolarizzatori.
La classe contiene due metodi, call e gradient, che
calcolano i termini da aggiungere rispettivamente
alla loss e alla regola di apprendimento per la 
regolarizzazione, che può essere L1 o L2
"""

class Regularizer:
    def __init__(self, labda):
        self.labda = labda
    
    def __call__(self, weights):
        raise NotImplementedError
    
    def gradient(self, weights):
        raise NotImplementedError

class L1(Regularizer):
    def __call__(self, weights):
        return self.labda * np.sum(np.abs(weights))
    
    def gradient(self, weights):
        return self.labda * np.sign(weights)

class L2(Regularizer):
    def __call__(self, weights):
        return self.labda * np.sum(weights**2)
    
    def gradient(self, weights):
        return 2 * self.labda * weights