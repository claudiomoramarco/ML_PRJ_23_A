import numpy as np

# per adesso solo MeanSquaredError


"""
Calcola la Mean Squared Error (MSE) tra le etichette reali e le previsioni.

Argomenti:
- y_true: Lista o array delle etichette reali.
- y_pred: Lista o array delle previsioni del modello.

Restituisce:
- MSE Loss.
"""
def mean_squared_error(y_true, y_pred):
    
    if len(y_true) != len(y_pred):
        raise ValueError("Le lunghezze di y_true e y_pred devono essere uguali.")

    N = len(y_true)
    mse = sum(np.sum((np.array(y_true[i]) - np.array(y_pred[i]))**2) for i in range(N)) / N
    return mse




def contributo_loss(y_true, y_pred, indice_output):
    """
    Calcola il contributo della loss rispetto a un singolo output.

    Argomenti:
    - y_true: Etichette reali.
    - y_pred: Previste del modello.
    - indice_output: Indice dell'output per il quale calcolare il contributo.

    Restituisce:
    - contributo: Contributo della loss rispetto a un singolo output.
    """
    contributo = 2 * (np.array(y_pred) - np.array(y_true)) / len(y_true)
    return contributo[indice_output]



# calcola la derivata della MeanSquaredError rispetto alle previsioni 
def backward(self, pred, targets):
    N = targets.size
    return 2 * (pred - targets) / N


