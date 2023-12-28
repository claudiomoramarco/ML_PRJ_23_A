import numpy as np
import activation_functions

class Unit:

    def __init__(self, activation_function, weigth_matrix, id, bias): # qua aggiungere il bias
        # weigths è la riga della matrice dei pesi corrispondente al neurone considerato
        self.activation_function = activation_function
        self.bias = bias
        # self.bias = bias
        self.weigth_matrix = weigth_matrix
        self.id = id
 

    def output_perceptron(self, value, id_prec):
        if id_prec == -1:
            out = value
        else:
            out = value * self.weigth_matrix[id_prec][self.id] + self.bias 
        return out


# non considera ancora la funzione di attivazione
class Layer:

    # n_prec è il numero di neuroni dello strato precedente
    # cont_units è il numero del primo neurone da creare
    def __init__(self, isHidden, isInput, n_units, weigth_matrix, cont_units, precIsHidden, n_prec): 
         
        self.n_units = n_units
        self.isHidden = isHidden # true se è hidden layer
        self.isInput = isInput # true se è input layer
        self.precIsHidden = precIsHidden # true se lo strato precedente è hidden 
        self.n_prec = n_prec
        # # GENERAZIONE bias, da rivedere
        # # numero pesi pari al numero di neuroni dello strato 
        # if isInput:
        #     self.bias = np.ones(self.n_units) # tutti 1 se è input layer
        # else: 
        #     self.bias = np.random(self.n_units) # creati inizialmente a caso


        # inizializzazione i suoi neuroni 
        self.units = []
        for i in range(self.n_units):
            # lineare per input e output
            # id+i è il numero del neurone che si crea
            if not isHidden:
                unit_tmp = Unit(activation_functions.linear, weigth_matrix, cont_units+i, np.random.randn())
            # ReLu per gli strati nascosti
            else:
                unit_tmp = Unit(activation_functions.relu, weigth_matrix, cont_units+i, np.random.randn() )
            self.units.append(unit_tmp)

        



    def get_nUnits(self):
        return self.n_units


    # nelle prime n_prec entries di inputs ci stanno i valori per il primo neurone dello strato
    def run(self, inputs):
        out = []

        for unit in self.units:
            for i in range(len(inputs)): # ogni entry dell'input 
                if not self.isInput:
                    output = unit.output_perceptron(inputs[i], unit.id - self.n_prec + i )
                else: 
                    output = unit.output_perceptron(inputs[i], -1 )
            out.append(np.sum(output))

        return out    
    






