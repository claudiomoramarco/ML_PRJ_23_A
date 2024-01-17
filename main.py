import read_data
import activation_functions
import nn
import numpy as np
import pickle # per salvare la rete
import sys
import matplotlib.pyplot as plt # visualizzazione

# python3 main.py <isTraining> <isNew> <isClassification> <epochsNumber> <filename> <filenameOldNetwork>

if len(sys.argv) < 5:
    print("È necessario un argomento")
    exit()

else:
    
    isTraining = sys.argv[1]
    network_is_new = sys.argv[2] # 1 per creare nuova rete, altrimenti si legge da file
    isClassification = sys.argv[3]
    numberEpochs = int(sys.argv[4]) # numero epoche per TR
    filename = sys.argv[5]
    filenameOldNetwork = sys.argv[6]

    if isTraining == '1':
        # lettura TR
        training_set = read_data.read_set(filename)

        if isClassification == '0':
            
            # NORMALIZZAZIONE DEI DATI
            training_set = np.array(training_set)
            training_set = training_set.astype(float)
            mean_values = np.mean(training_set, axis=0) #mean_values e std_dev_values servono dopo per il TS
            std_dev_values = np.std(training_set, axis=0)
            training_set = read_data.normalize_data(training_set, std_dev_values, mean_values)

            # divido in id , input , target dei TR e salvo nelle mie strutture dati
            # ids = [] # lista degli id del TR => non serve a niente

            inputs = [] # lista degli array di input 
            targets = [] #lista degli array target
            for riga in training_set:
                inputs.append(list(map(float, riga[0:-3])))
                targets.append(list(map(float, riga[-3:])))

            # se leggo i valori della rete da file
            if network_is_new == '0':
                try:
                    # Per caricare l'istanza della rete da un file
                    with open(filenameOldNetwork, 'rb') as file:
                        network_instance = pickle.load(file)
            
                except FileNotFoundError:
                    print("Il file non esiste.")
                    exit()
            # rete nuova
            else:
                network_instance = nn.NN(len(inputs[0]), len(targets[0]), 15, 1, 0.002, activation_functions.relu, 0)
            
            meanerrors = network_instance.run_training(inputs,targets,numberEpochs)
            # print(meanerrors)
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

        else: # isClassification
            # lettura TR
            training_set = read_data.read_forClassification(filename)
            inputs = training_set[0]
            targets = training_set[1]
            if network_is_new == '0':
                print("Aggiungere lettura old network")
            else: 
                # targets = [[value] for value in targets]
                # network_instance = nn.NN(len(inputs[0]), 1 , 15 , 1 , 0.2 , activation_functions.sigmoid, 1)
                network_instance = nn.NN(len(inputs[0]), 1 , 15 , 1 , 0.2 , activation_functions.sigmoid, 1)
                meanerrors = network_instance.run_training(inputs,targets,numberEpochs)
                # print(meanerrors)
                # STAMPO GRAFICO DELLA MSE 
                # Lista del numero di epoche
                epochs = list(range(1, len(meanerrors) + 1))
                # Creazione del grafico
                plt.plot(epochs, meanerrors, marker='o', linestyle='-', color='r')
                plt.title('Percentuale di output corretti rispetto al numero di epoche')
                plt.xlabel('Numero di epoche')
                plt.ylabel('%')
                plt.grid(True)
                plt.show()



    else: # isTest 
        print("Test")



    


#########################################################################################################################################


# # TEST SET
# # lettura TS 
# if test == '1':

#     test_set = read_data.read_set("ML-CUP/ML-CUP23-TS.csv")

#     # NORMALIZZAZIONE dei dati sulla base dei parametri del TR  => qui il problema è il numero non coerente di valori in input
#     test_set = np.array(test_set)
#     test_set = test_set.astype(float)
#     mean_values = np.mean(test_set, axis=0) #mean_values e std_dev_values servono dopo per il TS
#     std_dev_values = np.std(test_set, axis=0)
#     normalized_ts = read_data.normalize_data(test_set, std_dev_values, mean_values)
#     # parsing dell'input
#     test_inputs = [] # lista degli array di input 
#     test_targets = [] #lista degli array target
#     for riga in test_set:
#         test_inputs.append(list(map(float, riga[0:-3])))
#         test_targets.append(list(map(float, riga[-3:])))

#     out = network_instance.run_test(test_inputs, test_targets)

#     # RITORNO DAI DATI NORMALIZZATI AI DATI ORIGINALI (DELLE ULTIME TRE COLONNE)
#     original_data = out * std_dev_values[-3, -2, -1] + mean_values[-3, -2, -1]  

#     print(original_data)