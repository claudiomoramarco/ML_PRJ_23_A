import numpy as np
import activation_functions
import loss_functions
import layers


class Network: 

    # inizializzo con i valori del TR
    def __init__(self, inputs, targets, n_Hlayers, n_units):
        # genero i pesi iniziali a caso 
        # self.weigths = np.random.randn(10)
        self.tr_data = inputs
        self.n_Hlayers = n_Hlayers
        self.n_units = n_units # numero neuroni negli hidden layers
        self.tr_targets = targets
        self.last_predictions = [] # mantiene gli ultimi output calcolati
        self.n_inputUnits = len(self.tr_data[0]) #numero di neuroni input pari al numero di valori negli input
        self.n_outputUnits = len(self.tr_targets[0]) #numero di neuroni output pari al numero di valori nei target     
        
        # genero un peso casuale per ogni collegamento
        n_weights = self.n_inputUnits * self.n_Hlayers + self.n_Hlayers * self.n_outputUnits + pow(self.n_Hlayers,2)*(n_Hlayers-1)  
        self.weights = np.random(n_weights)

        # creazione dei LAYERS
        self.layers = []
        # input layer 
        self.layers.append(layers.Layer(0, len(self.tr_data[0]), 0))
        # hidden layers
        for i in range(self.n_Hlayers):
            if i == 0:
                self.layers.append(layers.Layer(1, len(self.tr_data[0])*2, self.layers[0].get_nUnits()))
            else:
                self.layers.append(layers.Layer(1, len(self.tr_data[0])*2, len(self.tr_data[0])*2))
        # output layer 
        self.layers.append(layers.Layer(0, len(self.tr_targets[0]), len(self.tr_data[0])*2))


    # calcola PER ORA SENZA ADDESTRAMENTO l'output con i pesi casuali per ogni riga di input del TR
    def run(self):
        