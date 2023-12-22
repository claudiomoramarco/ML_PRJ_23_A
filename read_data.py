import csv
from itertools import islice

# per adesso legge solo il TR
def read_TR():

    training_set = []
    numero_di_riga_iniziale = 7  # controllare che non ne salti una 

    # with è utilizzato per aprire un file e garantire che venga chiuso correttamente alla fine, 
    # anche se si verifica un'eccezione nel blocco di codice.
    with open('ML-CUP23-TR.csv', 'r') as file:
        
        # Creare un lettore CSV
        lettore_csv = csv.reader(file)
        # Leggi tutte le righe dal file CSV
        tutte_le_righe = list(lettore_csv)
        # Leggi solo le righe dalla posizione desiderata in avanti
        righe_selezionate = tutte_le_righe[numero_di_riga_iniziale:]

        for riga in righe_selezionate:
            training_set.append(riga)

    return training_set



        



