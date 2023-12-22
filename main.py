import math
import read_data

# TO DO : 

TR = read_data.read_TR()

# divido in id , input , target 
ids = [] # lista degli id del TR
inputs = [] # lista degli array di input 
targets = [] #lista degli array target

for riga in TR:
    ids.append(int(riga[0]))
    inputs.append(list(map(float, riga[1:-3])))
    targets.append(list(map(float, riga[-3:])))


