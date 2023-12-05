import numpy as np
import activationfunctions as act
import regularization as reg

"""Questa classe definisce un layer completamente connesso"""

class Layer:

    def __init__(self, units, input_shape, activation_function, 
                 bias=0.5, regularizer=None, initializer="uniform"):
        self.units = units #numero di unità del layer
        self.input_shape = input_shape #dimensione dell'input
        self.activation = activation_function #funzione di attivazione dei neuroni nel layer
        self.regularizer = regularizer #metodo di regolarizzazione
        self.bias = bias #bias iniziale
        self.W, self.b = self.initialize_weights(initializer) #inizializzazione dei pesi
        self.last_input = self.last_output = self.last_delta = None #inizializzazione di input, output e gradiente

        """
        Definiamo la funzione che inizializza i pesi.
        I pesi sono inizializzati usando una distribuzione uniforme nell'intervallo
        [-limit,+limit], dove limit è scelto dal parametro inizializer.
        L'inizializzazione Xavier (o Glorot) è particolarmente utile per le reti
        neurali con funzioni di attivazione simmetriche come il tanh.
        Aiuta a mantenere la varianza dei dati in entrata e in uscita dallo strato.
        L'inizializzazione He è ottimale per reti con funzioni di attivazione ReLU
        e varianti, in quanto aiuta a mitigare il problema dei gradienti che si
        estinguono o esplodono nelle fasi iniziali dell'addestramento.
        In questa funzione inseriamo anche l'inizializzazione del gradiente per il momento.
        """    

    def initialize_weights(self, initializer):
        limit = {'uniform': 0.5, 'xavier': np.sqrt(1 / self.input_shape), 'he': np.sqrt(2 / self.input_shape)}.get(initializer, 0.5)
        W = np.random.uniform(-limit, limit, (self.input_shape, self.units))
        b = np.random.uniform(-limit, limit, (self.units, 1))
        self.last_deltaw = np.zeros(self.W.shape)
        return W, b
    
    """
    Definiamo la funzione che calcola l'output di uno strato della rete dato un input. Il parametro x rappresenta i dati in input per lo strato corrente.
    net calcola l'input lineare allo strato, pesando gli input x con la matrice (trasposta) dei pesi W dello strato.
    activation applica all'input la funzione di attivazione e otteniamo in uscita l'output delo strato.
    """
    
    def forward(self, x):
        self.last_input = x
        self.last_net = np.dot(self.W.T, x) + self.b
        self.last_output = self.activation.activation(self.last_net)
        return self.last_output
    
    """
    Definiamo la funzione che fa la backpropagation per l'aggiornamento dei pesi.
    I parametri sono quattro: output_error, che calcola il gradiente dell'errore rispetto al net,
    eta che è il tasso di apprendimento, alfa che indica il peso da dare al momento, nesterov, un parametro intero che indica se
    il momento da usare è quello di nesterov (1) o no (0)
    """
    
    
    def backward(self, output_error, eta, alpha=0.0, nesterov=False):
        if nesterov == True:
            look_ahead_W = self.W - alpha * self.last_deltaw
        else:
            look_ahead_W = self.W
        
        delta = output_error * self.activation.derivative(np.dot(look_ahead_W.T, self.last_input) + self.b)
        
        momentum_term = alpha * self.last_deltaw
        
        # Calculate il gradiente
        delta_w = np.dot(self.last_input, delta.T) + momentum_term
        
        # Aggiorniamo pesi e bias
        self.W -= eta * (delta_w + self.regularize())
        self.b -= eta * delta
        
        # Aggiorniamo l'ultimo gradiente per la successiva iterazione del momento
        self.last_deltaw = delta_w
        
        # Otteniamo in uscita il gradiente per il layer precedente
        return np.dot(self.W, delta)
    
    def regularize(self):
        return self.regularizer.gradient(self.W) if self.regularizer else 0
    
    def __str__(self):
        return f"Weights: {self.W}\nBiases: {self.b}"

class InputLayer(Layer):
    """Input layer."""

    def __init__(self, units: int) -> None:
        self.output_shape = units
        self.units = units

    def output(self, x):
        return x