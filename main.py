import read_data
import activation_functions
import nn
import numpy as np
import pickle # per salvare la rete
import sys
import matplotlib.pyplot as plt # visualizzazione

# python3 main.py 1 (PER NUOVA RETE)
# python3 main.py 0 (PER RIPRENDERE RETE SALVATA)
# python3 main.py <isNew> <numeroEpoche> <tryOnTestSet> 

if len(sys.argv) < 5:
    print("È necessario un argomento")
    exit()

else:

    network_is_new = sys.argv[1] # 1 per creare nuova rete, altrimenti si legge da file
    numberEpochs = int(sys.argv[2]) # numero epoche per TR
    test = sys.argv[3] # 1 per TS, 0 altrimenti 

    # lettura TR
    training_set = read_data.read_set("ML-CUP/ML-CUP23-TR.csv")

    # NORMALIZZAZIONE DEI DATI
    training_set = np.array(training_set)
    training_set = training_set.astype(float)
    mean_values = np.mean(training_set, axis=0) #mean_values e std_dev_values servono dopo per il TS
    std_dev_values = np.std(training_set, axis=0)
    normalized_tr = read_data.normalize_data(training_set, std_dev_values, mean_values)


    # divido in id , input , target dei TR e salvo nelle mie strutture dati
    # ids = [] # lista degli id del TR => non serve a niente

    inputs = [] # lista degli array di input 
    targets = [] #lista degli array target
    for riga in normalized_tr:
        inputs.append(list(map(float, riga[0:-3])))
        targets.append(list(map(float, riga[-3:])))

    # se leggo i valori della rete da file
    if network_is_new == '0':
        try:
            # Per caricare l'istanza della rete da un file
            with open('rete.pkl', 'rb') as file:
                network_instance = pickle.load(file)
    
        except FileNotFoundError:
            print("Il file 'rete.pkl' non esiste.")
            exit()
    
    # rete nuova
    else:
        network_instance = nn.NN(len(inputs[0]), len(targets[0]), 20, 2, 0.001, activation_functions.relu)
    
    
    meanerrors = network_instance.run_training(inputs,targets,numberEpochs)

    # Salva l'istanza della rete su un file => NON SO SE FUNZIONA
    with open('rete.pkl', 'wb') as file:
        pickle.dump(network_instance, file)


    # STAMPO GRAFICO DELLA MSE 
    # Lista del numero di epoche
    epochs = list(range(1, len(meanerrors) + 1))
    # Creazione del grafico
    plt.plot(epochs, meanerrors, marker='o', linestyle='-', color='b')
    plt.title('Mean Squared Error (MSE) rispetto al numero di epoche')
    plt.xlabel('Numero di epoche')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.show()


#########################################################################################################################################


# TEST SET
# lettura TS 
if test == '1':

    test_set = read_data.read_set("ML-CUP/ML-CUP23-TS.csv")

    # NORMALIZZAZIONE dei dati sulla base dei parametri del TR  => qui il problema è il numero non coerente di valori in input
    test_set = np.array(test_set)
    test_set = test_set.astype(float)
    mean_values = np.mean(test_set, axis=0) #mean_values e std_dev_values servono dopo per il TS
    std_dev_values = np.std(test_set, axis=0)
    normalized_ts = read_data.normalize_data(test_set, std_dev_values, mean_values)
    # parsing dell'input
    test_inputs = [] # lista degli array di input 
    test_targets = [] #lista degli array target
    for riga in test_set:
        test_inputs.append(list(map(float, riga[0:-3])))
        test_targets.append(list(map(float, riga[-3:])))

    out = network_instance.run_test(test_inputs, test_targets)

    # RITORNO DAI DATI NORMALIZZATI AI DATI ORIGINALI (DELLE ULTIME TRE COLONNE)
    original_data = out * std_dev_values[-3, -2, -1] + mean_values[-3, -2, -1]  

    print(original_data)