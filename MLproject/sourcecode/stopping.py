import numpy as np
from layers import Layer

class Callback:
    def __init__(self,*args,**kwargs):
        self.args = args
        self:kwargs = kwargs
    
    def __call__(self,*args,**kwargs):
        raise NotImplementedError("Questo metodo deve essere implementato nella sottoclasse.")

class EarlyStopping(Callback):
    """La classe definisce i criteri di early stopping."""
    """
    __init__ inizializza i criteri di early stopping con un valore di
    patience (numero di epoche da attendere prima di fermarsi) e
    verbose (se impostato su True, stamperà messaggi quando l'arresto
    anticipato si attiva).
    Argomenti:
        patience (float): Numero di epoch in cui non ci sono miglioramenti prima dello stop.
        verbose (boolean): Determina se stampare messaggi quando l'arresto anticipato si attiva.
        counter: Un contatore per tenere traccia delle epoche senza miglioramenti.
        best_loss: La migliore perdita registrata finora, inizializzata a "infinito".
        best_weights e best_biases: Liste vuote per memorizzare i migliori pesi e bias del modello.
    """
    def __init__(self, patience=10, verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.infty
        self.best_weights = []
        self.best_biases = []
    
    """
    __call__ viene chiamato durante l'addestramento della rete.
    Controlla se il valore di perdita attuale della rete è inferiore
    al miglior valore di perdita registrato. Se sì, aggiorna il
    miglior valore di perdita e resetta il contatore. Inoltre,
    memorizza i pesi e i bias correnti come i migliori.
    Se il miglior valore di perdita non migliora per un numero di
    epoche pari a patience, interrompe l'addestramento e ripristina
    i pesi e i bias alla loro condizione ottimale.
    Argomenti:
        patience (float): Numero di epoch prima dello stop.
    """
    def __call__(self, model):
        """
        Se la loss migliora, vengono aggiornati il valore migliore
        della perdita, i pesi e i bias migliori, e il contatore
        viene azzerato. Se non lo fa, il contatore viene incrementato.
        """
        if model.get_loss_value() < self.best_loss:
            self.best_loss = model.get_loss_value()
            self.best_weights = model.get_weights()
            self.best_biases = model.get_biases()
            self.counter = 0
            
            for layer in model.layers[1:]:
                self.best_weights.append(layer.get_weights()[0])
                self.best_biases.append(layer.get_biases()[0])
        
        else:
            self.counter += 1
        
        """
        Se il contatore supera la patience, si ferma l'addestramento.
        Se verbose è attivo, allo stop dell'addestramento vengono 
        stampati i valori della loss, dei pesi e dei bias migliori.
        """
        if self.counter >= self.patience:
            model.training = False

            for i, layer in enumerate(model.layers[1:]):
                layer.W = self.best_weights[i]
                layer.bias = self.best_biases[i]

            if self.verbose:
                print("Early stopping")
                print("Best loss: {:.4f}".format(self.best_loss))
                print("Best weights: {}".format(layer.W))
                print("Best biases: {}".format(layer.b))

            return True