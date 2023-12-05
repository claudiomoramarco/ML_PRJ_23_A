import numpy as np

class Loss:
    def loss(self, pred, labels):
        """Calcola il valore della funzione di perdita.
        
        Argomenti:
            pred (np.array): Predizioni del modello.
            labels (np.array): Etichette/Labels veri.
        
        Raises:
            NotImplementedError: Se il metodo non è implementato nella sottoclasse.
        """
        raise NotImplementedError("Questo metodo deve essere implementato nella sottoclasse.")

    def backward(self, pred, labels):
        """Calcola il gradiente della funzione di perdita.
        
        Args:
            pred (np.array): Predizioni del modello.
            labels (np.array): Etichette/Labels veri.
        
        Raises:
            NotImplementedError: Se il metodo non è implementato nella sottoclasse.
        """
        raise NotImplementedError("Questo metodo deve essere implementato nella sottoclasse.")

class MeanSquaredError(Loss):
    def loss(self, pred, labels):
        """Calcola la media degli errori quadratici.
        
        Argomenti:
            pred (np.array): Predizioni del modello.
            labels (np.array): Etichette/Labels veri.
        
        Output:
            float: Valore medio dell'errore quadratico.
        """
        return np.mean(np.square(pred - labels))

    def backward(self, pred, labels):
        """Calcola la derivata della MSE rispetto alle previsioni.
        
        Argomenti:
            pred (np.array): Predizioni del modello.
            labels (np.array): Etichette/Labels veri.
        
        Output:
            np.array: Gradiente della funzione di perdita.
        """
        N = labels.size
        return 2 * (pred - labels) / N

class BinaryCrossEntropy(Loss):
    def loss(self, pred, labels):
        """Calcola l'entropia incrociata binaria.
        
        Args:
            pred (np.array): Predizioni del modello.
            labels (np.array): Etichette/Labels veri.
        
        Returns:
            float: Valore dell'entropia incrociata binaria.
        """
        return -(labels * np.log(pred) + (1 - labels) * np.log(1 - pred)).mean()

    def backward(self, pred, labels):
        """Calcola la derivata della BCE rispetto alle previsioni.
        
        Args:
            pred (np.array): Predizioni del modello.
            labels (np.array): Etichette/Labels veri.
        
        Returns:
            np.array: Gradiente della funzione di perdita.
        """
        return - (labels / pred) + ((1 - labels) / (1 - pred))
