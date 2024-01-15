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


        
def normalize_data(data_set, std_dev_values, mean_values ):
    normalized_data = (data_set - mean_values) / std_dev_values
    return normalized_data


