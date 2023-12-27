import math
import read_data
import activation_functions
import neural_network
from neural_network import Network
import numpy as np

# TO DO : 
# - controlla attivazione  
# - controlla derivata di softmax 

# Per ora ho fatto un solo layer con 3 neuroni (numero di valori target)

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


# test rete fatta fin ora 
network_instance = neural_network.Network(inputs,targets,1,20) # 1 hidden layer con 20 neuroni
output = network_instance.run()
print(output)

