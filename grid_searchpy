import random
import itertools

def genera_layers():
    start = 20
    end = 150

    # Assegna pesi per le diverse lunghezze
    lunghezza_possibile = [1, 2, 3]
    pesi = [1, 3, 5] 

    # Genera un numero casuale per determinare la lunghezza della sequenza
    lunghezza_sequenza = random.choices(lunghezza_possibile, weights=pesi)[0]

    # Genera la sequenza di lunghezza casuale
    sequenza = [random.randint(start, end) for _ in range(lunghezza_sequenza)]

    # Se la sequenza ha lunghezza 1, deve essere maggiore di 100 e divisibile per 5
    if lunghezza_sequenza == 1:
        sequenza[0] = random.randint(max(101, start), end)
        sequenza[0] = (sequenza[0] // 5) * 5  # Rendi il valore divisibile per 5

    # Se la sequenza ha lunghezza 2, assicurati che entrambi i valori siano maggiori di 50,
    # divisibili per 5 e la differenza non sia maggiore di 20
    elif lunghezza_sequenza == 2:
        sequenza[0] = random.randint(max(51, start), end)
        sequenza[0] = (sequenza[0] // 5) * 5  # Rendi il valore divisibile per 5

        # Correggi il calcolo della differenza tra i valori
        max_val = min(sequenza[0] + 20, end)
        sequenza[1] = random.randint(max(sequenza[0] + 5, start), max_val)
        sequenza[1] = (sequenza[1] // 5) * 5  # Rendi il valore divisibile per 5

    # Se la sequenza ha lunghezza 3, assicurati che tutti e tre i valori siano divisibili per 5
    # e la differenza tra i valori non sia maggiore di 20
    elif lunghezza_sequenza == 3:
        sequenza[0] = random.randint(max(51, start), end)
        sequenza[0] = (sequenza[0] // 5) * 5  # Rendi il valore divisibile per 5

        sequenza[1] = random.randint(max(sequenza[0] + 5, start), min(sequenza[0] + 20, end))
        sequenza[1] = (sequenza[1] // 5) * 5  # Rendi il valore divisibile per 5

        # Assicurati che l'intervallo sia valido prima di generare sequenza[2]
        min_val = max(sequenza[1] + 5, start)
        max_val = min(min_val + 20, end)
        
        if min_val <= max_val:
            sequenza[2] = random.randint(min_val, max_val)
            sequenza[2] = (sequenza[2] // 5) * 5  # Rendi il valore divisibile per 5
        else:
            # Se l'intervallo è vuoto, regenera i valori precedenti
            return genera_layers()

    return sequenza


def genera_float():
    cifre_decimali = random.choices([1, 2, 3, 4], weights=[10, 70, 70, 15])[0]
    valore = round(random.uniform(0.00001, 0.65), cifre_decimali)
    return valore


layers = []
learning_rate = []
momentum = []
l2_values = [0, 1]
reg_coefficient = []
batch_size = [2, 5, 10]

for i in range(10):
    layers.append(genera_layers())
    learning_rate.append(genera_float())
    momentum.append(genera_float())
    reg_coefficient.append(genera_float())

# Genera tutte le combinazioni
tutte_combinazioni = list(itertools.product(layers, learning_rate, momentum, l2_values, reg_coefficient, batch_size))

# Scrivi le combinazioni nel file
with open('combination_output.txt', 'w') as file:
    for combinazione in tutte_combinazioni:
        hl_str = ','.join(map(str, combinazione[0]))  # Converti gli hidden layers in una stringa
        line = f"[{hl_str}];{combinazione[1]};{combinazione[2]};{combinazione[3]};{combinazione[4]};{combinazione[5]}\n"
        file.write(line)

with open('combination_output.txt', 'r') as file:
    lines = file.readlines()

# Calcola il numero di righe da mantenere (30% delle righe originali)
lines_to_keep = int(len(lines) * 0.1)

# Seleziona casualmente le righe da mantenere
random_lines = random.sample(lines, lines_to_keep)

# Scrivi le righe selezionate in un nuovo file
with open('combination_output.txt', 'w') as file:
    file.writelines(random_lines)

print("Le combinazioni sono state salvate nel file: combination_output.txt")
