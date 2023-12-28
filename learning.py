import numpy as np
import loss_functions



def gradiente_rispetto_ai_pesi(weights, targets, output):
    # Calcolo della funzione di loss
    loss_value = loss_functions.mean_squared_error(targets, output)

    # Inizializzazione del gradiente rispetto ai pesi
    gradiente = 0

    # Numero di esempi nel set di dati
    N = len(output)

    for i in range(N):
        # Calcolo del contributo dell'output nella loss
        contributo_output = loss_functions.contributo_loss(targets, output, i)

        contributo_output = np.array(contributo_output)
        output[i] = np.array(output[i])
        # Aggiornamento del gradiente rispetto ai pesi
        gradiente += contributo_output * output[i]
    
    # Media del gradiente rispetto ai pesi su tutti gli esempi
    gradiente /= N

    return gradiente

