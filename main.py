import read_data
import activation_functions
import nn
import numpy as np
import pickle # per salvare la rete


# lettura TR
TR = read_data.read_TR()

# divido in id , input , target dei TR e salvo nelle mie strutture dati
# ids = [] # lista degli id del TR => non serve a niente
inputs = [] # lista degli array di input 
targets = [] #lista degli array target
for riga in TR:
    inputs.append(list(map(float, riga[0:-3])))
    targets.append(list(map(float, riga[-3:])))

network_instance = nn.NN(len(inputs[0]), len(targets[0]), 20, 5, 0.1, activation_functions.relu)
network_instance.run(inputs,targets,3)

# # Salva l'istanza della rete su un file
# with open('rete.pkl', 'wb') as file:
#     pickle.dump(network_instance, file)

