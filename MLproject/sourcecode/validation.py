from network import Network
from lossfunctions import MeanSquaredError
from metrics import BinaryAccuracy
from stopping import EarlyStopping
import numpy as np

def kfold_cv(model: Network, x, y, k=5, **kwargs):
    """
    Esegue la validazione incrociata a k-fold su un dataset.

    Parametri:
    model: Il modello di rete neurale da addestrare
    x: dataset di input
    y: dataset di output
    k: Numero di fold per la validazione incrociata
    kwargs: Altri argomenti per il metodo di addestramento

    Output:
    Dizionario con metriche medie di valutazione e perdita
    """
    if k < 1:
        raise ValueError("k deve essere maggiore di 1")

    # Divisione del dataset in k parti (fold)
    x_folds = np.array_split(x, k)
    y_folds = np.array_split(y, k)

    # Estrazione e configurazione dei parametri da kwargs con valori predefiniti
    metric = kwargs.get("metric", BinaryAccuracy())
    loss = kwargs.get("loss", MeanSquaredError())
    epochs = kwargs.get("epochs", 100)
    eta = kwargs.get("eta", 10e-3)
    callbacks = kwargs.get("callbacks", [EarlyStopping])
    patience = kwargs.get("patience", epochs / 100 * 5)
    verbose = kwargs.get("verbose", False)
    nesterov = kwargs.get("nesterov", False)
    scaler = kwargs.get("scaler", None)

    # Inizializzazione delle liste per memorizzare le metriche
    val_metrics = []
    tr_metrics = []
    losses = []
    val_losses = []

    for i in range(k):
        # Preparazione dei dataset di training e validazione per l'attuale fold
        x_train, y_train = np.concatenate(x_folds[:i] + x_folds[i+1:]), np.concatenate(y_folds[:i] + y_folds[i+1:])
        x_val, y_val = x_folds[i], y_folds[i]

        # Addestramento del modello
        model.train(
            (x_train, y_train),
            (x_val, y_val),
            metric=metric,
            loss=loss,
            epochs=epochs,
            eta=eta,
            verbose=verbose,
            nesterov=nesterov,
            callbacks=[callbacks[0](patience)]
        )

        # Calcolo delle metriche per il training e la validazione
        y_pred_val, y_pred_train = model.multiple_outputs(x_val), model.multiple_outputs(x_train)
        y_val_scaled, y_pred_val_scaled = (y_val, y_pred_val) if scaler is None else (scaler.inverse_transform(y_val), scaler.inverse_transform(y_pred_val))
        y_train_scaled, y_pred_train_scaled = (y_train, y_pred_train) if scaler is None else (scaler.inverse_transform(y_train), scaler.inverse_transform(y_pred_train))

        # Salvataggio delle metriche calcolate
        val_metrics.append(metric(y_pred_val_scaled, y_val_scaled))
        tr_metrics.append(metric(y_pred_train_scaled, y_train_scaled))
        val_losses.append(loss.loss(y_pred_val_scaled, y_val_scaled))
        losses.append(loss.loss(y_pred_train_scaled, y_train_scaled))

        # Reimpostazione dei pesi del modello
        model.reset_weights()

    # Calcolo delle medie delle metriche su tutti i fold e ritorno del risultato
    return {
        "val_metric": np.mean(val_metrics),
        "tr_metric": np.mean(tr_metrics),
        "losses": np.mean(losses),
        "val_losses": np.mean(val_losses)
    }

# Esempio di uso:
# network = Network(...)
# risultati = kfold_cv(network, x_data, y_data, k=10, epochs=50, eta=0.01, ...)
