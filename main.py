import read_data
import activation_functions
import nn


# lettura TR
TR = read_data.read_TR()

# divido in id , input , target dei TR e salvo nelle mie strutture dati
# ids = [] # lista degli id del TR
inputs = [] # lista degli array di input 
targets = [] #lista degli array target
for riga in TR:
    # ids.append(int(riga[0]))
    inputs.append(list(map(float, riga[1:-3])))
    targets.append(list(map(float, riga[-3:])))

network_instance = nn.NN(len(inputs[0]), len(targets[0]), 20, 2, 0.01, activation_functions.relu)
network_instance.run(inputs,targets,1)


