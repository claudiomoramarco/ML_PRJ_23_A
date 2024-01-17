import csv
import numpy as np

# from itertools import islice

# per adesso legge solo il TR
# legge l'intera riga
def read_set(filename):

    data_set = []
    numero_di_riga_iniziale = 7  # controllare che non ne salti una 

    # with è utilizzato per aprire un file e garantire che venga chiuso correttamente alla fine, 
    # anche se si verifica un'eccezione nel blocco di codice.
    with open(filename, 'r') as file:
        
        # Creare un lettore CSV
        lettore_csv = csv.reader(file)
        # Leggi tutte le righe dal file CSV
        tutte_le_righe = list(lettore_csv)
        # Leggi solo le righe dalla posizione desiderata in avanti
        righe_selezionate = tutte_le_righe[numero_di_riga_iniziale:]

        for riga in righe_selezionate:
            data_set.append(riga[1:])


    return data_set


def read_forClassification(filename):
    targets = []
    inputs = []
    with open(filename, 'r') as file:
        for line in file:
            # Dividi la riga in una lista di stringhe
            values = line.split()

            # Aggiungi il primo numero a "target"
            targets.append(int(values[0]))

            # Aggiungi i successivi valori a "inputs" come lista di interi
            input_features = list(map(int, values[1:-1]))
            inputs.append(input_features)
    

    tmp = []
    for inp in inputs:
        tmp.append(oneHotEncodingInput(inp)) 
    inputs = tmp
    tmp = []
  
    return(inputs, targets)



def normalize_data(data_set, std_dev_values, mean_values ):
    normalized_data = (data_set - mean_values) / std_dev_values
    return normalized_data



# input è una singola riga
def oneHotEncodingInput(input):
    if len(input) != 6:
        print("read_data:oneHotEncodingInput:ERROR")
        exit()
    encoded = []
    for i in range(len(input)):
        if i == 0 or i == 1 or i == 3:
            if input[i] == 1: 
                encoded+=[0,0,1]
            if input[i] == 2:
                encoded+=[0,1,0]
            if input[i] == 3:
                encoded+=[1,0,0]
        if i == 2 or i == 5:
            if input[i] == 1:
                encoded+=[0,1]
            if input[i] == 2:
                encoded+=[1,0]
        if i == 4:
            if input[i] == 1:
                encoded+=[0,0,0,1]
            if input[i] == 2:
                encoded+=[0,0,1,0]
            if input[i] == 3:
                encoded+=[0,1,0,0]
            if input[i] == 4:
                encoded+=[1,0,0,0]
    return encoded     

    

def oneHotEncodingTarget(t):
    if t == 0:
        return [0,1]
    return [1,0]

