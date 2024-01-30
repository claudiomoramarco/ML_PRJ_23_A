import random
from decimal import Decimal, getcontext

def genera_layers():
    start = 2
    end = 6
    while True:
        lunghezza_sequenza = 1
        sequenza = [random.randint(start, end) for _ in range(lunghezza_sequenza)]

        if all(s <= end for s in sequenza):
            break

    #aggiungi 1 a tutte le sequenze in terza posizione
    sequenza = sequenza + [1]
    return sequenza

def genera_float():
    cifre_decimali = random.choices([1, 2, 3, 4], weights=[10, 70, 70, 15])[0]
    current_precision = getcontext().prec
    getcontext().prec = cifre_decimali + 1
    min_value = Decimal('0.00001')
    max_value = Decimal('0.9')
    valore = min_value + (max_value - min_value) * Decimal(random.random())
    getcontext().prec = current_precision
    return valore

def genera_lr():
    cifre_decimali = random.choice([3, 4])
    numero = round(random.uniform(0.001, 0.09), cifre_decimali)
    numero_formattato = f"{numero:.{cifre_decimali}f}"
    return numero_formattato

def genera_momentum():
    cifre_decimali = random.choice([2, 3])
    numero = round(random.uniform(0.001, 0.5), cifre_decimali)
    return numero

num_combinations = 500

percent_to_hold = 10
num_combinations_to_keep = int(num_combinations * percent_to_hold / 10)

with open('combination_monks.txt', 'w') as file:
    for _ in range(num_combinations_to_keep):
        layers = genera_layers()
        learning_rate = genera_lr()
        momentum = genera_momentum()
        reg_coefficient = genera_lr()
        batch_size = random.choice([1, 2])

        hl_str = ','.join(map(str, layers))
        line = f"[{hl_str}];{learning_rate};{momentum};0;{reg_coefficient};{batch_size}\n"
        file.write(line)

print("Le combinazioni per i Monks sono state salvate nel file: combination_monks.txt")
