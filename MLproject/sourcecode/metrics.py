import numpy as np
from numpy import round, argmax
from sklearn.metrics import accuracy_score

"""Definiamo le funzioni di supporto per il calcolo dell'accuratezza."""

def binary_accuracy(y_pred, y_true, normalize=False):
    """Calcola l'accuratezza per la classificazione binaria.

    Argomenti:
        y_pred (array): Predizioni del modello, che vengono arrotondate all'intero più vicino.
        y_true (array): Etichette/Labels reali.

    Output:
        float: Accuratezza della classificazione.
    """
    y_pred = round(y_pred)
    return accuracy_score(y_true, y_pred)

def multiclass_accuracy(y_pred, y_true, normalize=False):
    """Calcola l'accuratezza per la classificazione multiclasse.

    Argomenti:
        y_pred (array): Predizioni del modello (probabilità per ciascuna classe).
        y_true (array): Etichette/Labels reali (codificate in one-hot).
    Convertiamo con argmax le probabilità predette e le etichette reali in classi
    predette e reali, poi calcoliamo l'accuratezza.
    
    Output:
        float: Accuratezza della classificazione.
    """
    y_pred = argmax(y_pred, axis=1)
    y_true = argmax(y_true, axis=1)
    return accuracy_score(y_true, y_pred)

"""Definiamo la classe base per le metriche"""

class Metric:
    def __init__(self, name=None):
        """Inizializza la metrica."""
        self.name = name

    def __call__(self, y_true, y_pred):
        """Metodo da implementare nelle sottoclassi per calcolare la metrica."""
        raise NotImplementedError("Questo metodo deve essere implementato nella sottoclasse.")

"""E ora definiamo le metriche specifiche con delle sottoclassi."""

class BinaryAccuracy(Metric):
    def __init__(self):
        """Inizializza la metrica di accuratezza binaria."""
        super().__init__(name="binary_accuracy")

    def __call__(self, y_pred, y_true):
        """Calcola l'accuratezza binaria."""
        return binary_accuracy(y_pred, y_true)

class MulticlassAccuracy(Metric):
    def __init__(self):
        """Inizializza la metrica di accuratezza multiclasse."""
        super().__init__(name="multiclass_accuracy")

    def __call__(self, y_pred, y_true):
        """Calcola l'accuratezza multiclasse."""
        return multiclass_accuracy(y_pred, y_true)

class MeanEuclideanError(Metric):
    def __init__(self):
        """Inizializza la metrica di errore euclideo medio."""
        super().__init__(name="mean_euclidean_error")

    def __call__(self, y_pred, y_true):
        """Calcola l'errore euclideo medio."""
        return np.mean(np.linalg.norm(y_pred - y_true, axis=1))