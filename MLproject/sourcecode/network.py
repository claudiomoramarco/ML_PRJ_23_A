import numpy as np
from typing import List
from tqdm import tqdm
import time
from layers import Layer, InputLayer
from activationfunctions import Activation
from metrics import Metric
from lossfunctions import Loss
from regularization import Regularizer
from stopping import Callback


""" Creiamo una barra di avanzamento che ci dica a che punto siamo"""
# Definiamo una variabile globale per il formato per la barra di avanzamento
fmt = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}[{postfix}]"

@staticmethod
def update_bar(bar, stats):
    """
    Aggiorna la barra di progresso.

    Parametri:
    bar: Barra di progresso da aggiornare.
    stats: Statistiche da mostrare nella barra di progresso.
    """
    bar.set_postfix(stats)
    bar.update(1)

class Network:
    """
    Your dense neural network class.
    """

    def __init__(self, input_shape: int, regularizer: Regularizer = None) -> None:
        """
        Inizializza la rete neurale.

        Parametri:
        input_shape: Numero di caratteristiche in input.
        regularizer: Regularizzatore da utilizzare (opzionale).
        """
        self.layers: List[Layer] = [InputLayer(input_shape)]
        self.tr_stats, self.val_stats, self.tr_err, self.val_err = [], [], [], []
        self.regularizer = regularizer
        self.bar = None
        self.training = True
    
    def add_layer(self, units, activation_function: Activation, bias=0.5, initializer: str = "uniform"):
        """
        Aggiunge un nuovo strato alla rete neurale.
    
        Parametri:

        units: Numero di unità nello strato.
        activation_function: Funzione di attivazione da utilizzare.
        bias: Bias iniziale (opzionale).
        initializer: Inizializzatore per pesi e bias (opzionale).
        """
        self.layers.append(Layer(units, self.layers[-1].output_shape, activation_function, bias, regularizer=self.regularizer, initializer=initializer))
    
    def __forward_prop__(self, x):
        """
        Esegue la propagazione in avanti di tutti gli strati della rete.
        
        Parametri:
        x: Input della rete.
        
        Output:
        Output della rete.
        """
        for layer in self.layers:
            y = layer.forward_prop(x)
        return y
    
    def __backward_prop__(self, delta, eta, nesterov):
        """
        Esegue la backpropagation durante l'addestramento.
        
        Parametri:
        delta: Differenza tra output previsto e reale.
        eta: Tasso di apprendimento.
        nesterov: Impostazione per il momento di Nesterov.
        """
        for layer in reversed(self.layers[1:]):
            delta = layer.update_weights(deltas=delta, eta=eta, nesterov=nesterov)
    
    def multiple_outputs(self, patterns):
        """
        Calcola gli output della rete per più input.
        
        Parametri:
        patterns: Insieme di input.
        
        Output:
        Output della rete per ogni input.
        """
        outputs = [self.__forward_prop__(p.reshape(len(p), 1)) for p in patterns]
        return np.array(outputs)
    
    def output(self, x):
        """
        Calcola l'output della rete per un singolo input.
        
        Parametri:
        x: Input della rete.
        
        Output:
        Output della rete.
        """
        return self.__forward_prop__(x)
    
    def train(self, train: tuple, validation: tuple, metric: Metric, loss: Loss, epochs=25, eta=10e-3, verbose=True, callbacks: List[Callback] = [], nesterov=0):
        """
        Addestra la rete neurale.
        
        Parametri:
        train: Coppia di dati e etichette di addestramento.
        validation: Coppia di dati e etichette di validazione.
        metric: Metrica di valutazione.
        loss: Funzione di loss.
        epochs: Numero di epoche di addestramento.
        eta: Tasso di apprendimento.
        verbose: Se vero, mostra la barra di progresso.
        callbacks: Lista di callback da utilizzare durante l'addestramento.
        nesterov: Momento di Nesterov.
        """
        train_data, train_labels = train
        val_data, val_labels = validation
        
        if verbose:
            self.bar = tqdm(total=epochs, desc="Training", bar_format=fmt)
            
        for epoch in range(epochs):
            if not self.training:
                break
            
            for x, target in zip(train_data, train_labels):
                pred = self.__forward_prop__(x)
                deltas = loss.backward(pred, target)
                self.__backward_prop__(deltas, eta, nesterov)
            
            self.epoch_stats(epoch, train_data, train_labels, val_data, val_labels, metric, loss, verbose, self.bar)
            
            for callback in callbacks:
                callback(self)
        
        stats = {
            "train_loss": self.tr_stats,
            "val_loss": self.val_stats,
            "train_acc": self.tr_err,
            "val_acc": self.val_err}
        
        if verbose:
            self.bar.close()
        
        return stats
    
    def epoch_stats(self, epoch, tr, tr_labels, val, val_labels, metric, loss, verbose, bar):
        """
        Calcola e registra le statistiche di perdita e accuratezza per ogni epoca.
        
        Parametri:
        epoch: Numero dell'epoca corrente.
        tr: Dati di allenamento.
        tr_labels: Etichette di allenamento.
        val: Dati di validazione.
        val_labels: Etichette di validazione.
        metric: Metrica di valutazione.
        loss: Funzione di perdita.
        verbose: Se vero, mostra la barra di progresso.
        bar: Barra di progresso.
        """
        tr_loss, tr_metric = loss.loss(self.multiple_outputs(tr), tr_labels), metric(self.multiple_outputs(tr), tr_labels)
        self.tr_stats.append(tr_loss)
        self.tr_err.append(tr_metric)
        
        val_loss = val_metric = None
        if val is not None:
            val_loss, val_metric = loss.loss(self.multiple_outputs(val), val_labels), metric(self.multiple_outputs(val), val_labels)
            self.val_stats.append(val_loss)
            self.val_err.append(val_metric)
        
        epoch_stats = {"loss": tr_loss, "val_loss": val_loss, "val_acc": val_metric}
        if verbose:
            update_bar(bar, epoch_stats)
    
    def get_loss_value(self):
        """
        Restituisce l'ultimo valore di perdita registrato.
        """
        return self.val_stats[-1] if self.val_stats else None
    
    def reset_weights(self):
        """
        Reimposta i pesi e i bias di ogni strato della rete.
        """
        self.val_stats = self.tr_stats = self.val_err = self.tr_err = []
        self.training = True
        
        for layer in self.layers[1:]:
            layer.init_layer()
