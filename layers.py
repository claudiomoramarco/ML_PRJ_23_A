import numpy as np
import activation_functions

class Unit:

    def __init__(self, activation_function):
        self.activation_function = activation_function

    def output(self, inputs):
        value = np.sum(inputs)
        return value


# non considera i pesi ma solo le funzioni di attivazione
class Layer:

    # n_prec è il numero di neuroni dello strato precedente
    def __init__(self, isHidden, n_units, n_prec): 
        self.n_units = n_units
        self.n_prec = n_prec
        self.isHidden = isHidden
        # inizializzare i suoi neuroni 
        self.units = []
        for i in range(self.n_units):
            if not isHidden:
                unit_tmp = Unit(activation_functions.linear)
            else:
                unit_tmp = Unit(activation_functions.sigmoid)
            self.units.append(unit_tmp)

    def get_nUnits(self):
        return self.n_units

    # nelle prime n_prec entries di inputs ci stanno i valori per il primo neurone dello strato
    def run(self, inputs):
        out = []
        if self.isHidden:
            for i in range(len(input)):
                out.append(self.units[i].output(inputs[i]))
        else:
            for i in range(self.n_units): # per ogni neurone si leggono i valori input corrispondenti e si calcola l'output
                inputs_tmp = []
                for j in range(self.n_prec):
                    inputs_tmp.append(inputs[i+j])
                out.append(self.units[i].output(inputs_tmp))
        return out    






