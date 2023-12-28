import numpy as np
import activation_functions
import layers
import learning

class Network: 

    # inizializzo con i valori del TR
    def __init__(self, inputs, targets, n_Hlayers, n_Hunits):
        # genero i pesi iniziali a caso 
        # self.weigths = np.random.randn(10)
        self.tr_data = inputs
        self.n_Hlayers = n_Hlayers
        self.n_Hunits = n_Hunits # numero neuroni negli hidden layers
        self.tr_targets = targets
        self.last_predictions = [] # mantiene gli ultimi output calcolati
        self.n_inputUnits = len(self.tr_data[0]) #numero di neuroni input pari al numero di valori negli input
        self.n_outputUnits = len(self.tr_targets[0]) #numero di neuroni output pari al numero di valori nei target     
        
        # GENERAZIONE matrice di PESI casuale
        self.weigths = np.random.randn(self.n_Hunits + self.n_inputUnits + self.n_outputUnits,self.n_Hunits + self.n_inputUnits + self.n_outputUnits)

        # creazione dei LAYERS

        cont_units = 0 # contatore neuroni creati

        self.layers = []
        # input layer 
        self.layers.append(layers.Layer(0, 1,  len(self.tr_data[0]), self.weigths, 0, 0, 0))
        cont_units += len(self.tr_data[0])

        # hidden layers
        for i in range(self.n_Hlayers):
            if i == 0:
                self.layers.append(layers.Layer(1, 0,  len(self.tr_data[0])*2, self.weigths, cont_units, 0, self.layers[0].get_nUnits()))
                cont_units += len(self.tr_data[0])*2
            else:
                self.layers.append(layers.Layer(1, 0,  len(self.tr_data[0])*2, self.weigths, cont_units, 1, len(self.tr_data[0])*2, ))
                cont_units += len(self.tr_data[0])*2
        
        # output layer 
        self.layers.append(layers.Layer(0, 0,  len(self.tr_targets[0]), self.weigths, cont_units, 1, len(self.tr_data[0])*2))


    # Per ora senza addestramento: 
        # calcola l'output 
        # calcola il gradiente rispetto ai pesi  
    def run(self):
        output = []
        #  ogni esempio TR
        for data in self.tr_data:
            data_tmp = self.tr_data
            # passa per ogni livello 
            for layer in self.layers:
                data_tmp = layer.run(data_tmp)
            
            output.append(data_tmp)
        
        # ora ho l'output dell'intero TR
        # calcolo il gradiente 
        return learning.gradiente_rispetto_ai_pesi(self.weigths, self.tr_targets, output)
