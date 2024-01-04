import read_data
# import activation_functions
# import neural_network
import numpy as np
# import learning
import nn_new

# TO DO : 
# - controlla attivazione  
# - controlla derivata di softmax 
# calcola il gradiente, ma viene un vettore di 3 elementi , capire cosa devo usare per l'aggiornamento dei pesi

# lettura TR
TR = read_data.read_TR()

# divido in id , input , target dei TR e salvo nelle mie strutture dati
ids = [] # lista degli id del TR
inputs = [] # lista degli array di input 
targets = [] #lista degli array target
for riga in TR:
    ids.append(int(riga[0]))
    inputs.append(list(map(float, riga[1:-3])))
    targets.append(list(map(float, riga[-3:])))



# # calcolo gradiente su tutto il TR senza addestramento 
# network_instance = neural_network.Network(inputs,targets,1,20) # 1 hidden layer con 20 neuroni
# network_instance.run_learning()

# versione nuova 
network_instance = nn_new.NN(len(inputs[0]), len(targets[0]), 20, 1)
network_instance.run(inputs,targets,1)


