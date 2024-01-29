import numpy as np
import re


def indexing(filename):
    # Leggi il contenuto del file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Inizializza un contatore per il nuovo indice
    new_index = 0

    # Apre un nuovo file per scrivere l'output
    with open(filename, 'w') as file:
        for line in lines:
            # Utilizza una espressione regolare per cercare il numero all'inizio di ogni riga
            match = re.match(r'^\d+', line)
            if match:
                # Estrai il numero e sostituiscilo con il nuovo indice
                old_index = int(match.group())
                line = line.replace(str(old_index), str(new_index), 1)
                new_index += 1

            # Scrivi la riga modificata nel nuovo file
            file.write(line)


def deleteLines(filename):
    # Leggi il contenuto del file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Filtra le righe con learning rate uguale a 0.0
    filtered_lines = [line for line in lines if "'learning_rate': 0.0," not in line]

    # Scrivi il risultato nel nuovo file
    with open(filename, 'w') as file:
        file.writelines(filtered_lines)




def findMin(filename):

    # Leggi il contenuto del file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Inizializza le variabili per tracciare il valore minimo e l'indice associato
    min_loss = float('inf')
    min_loss_index = None

    # Itera attraverso le righe e cerca la Loss minima
    for line in lines:
        # Estrai la Loss dalla riga
        loss = float(line.split(': Loss ')[1])

        # Aggiorna il valore minimo e l'indice associato se la Loss è minore
        if loss < min_loss:
            min_loss = loss
            min_loss_index = int(line.split(':')[0])

    # Stampa l'indice associato alla Loss minima
    print(f"L'indice associato alla Loss minima ({min_loss}) è: {min_loss_index}")



findMin('avg_loss_validation.txt')