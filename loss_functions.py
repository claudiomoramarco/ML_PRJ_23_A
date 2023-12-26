import numpy as np

# per adesso solo MeanSquaredError

class MeanSquaredError:

    # calcola la media degli errori quadratici
    # pred sono le predizioni del modello 
    def loss(self, pred, targets):
        if len(pred) != len(targets):
            raise ValueError("Le liste di predizioni e target devono avere la stessa lunghezza.")
        # Calcola la somma dei quadrati delle differenze per ciascuna entry
        mse = sum((p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2 + (p[2] - t[2]) ** 2 for p, t in zip(pred, targets))
        # Calcola la media dividendo per il numero totale di elementi
        mse /= len(pred)
        return mse
    
    # calcola la derivata della MeanSquaredError rispetto alle previsioni 
    def backward(self, pred, targets):
        N = targets.size
        return 2 * (pred - targets) / N
    

