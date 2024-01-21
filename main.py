import read_data
import activation_functions
import nn
import numpy as np
import pickle # per salvare la rete
import sys
import matplotlib.pyplot as plt # visualizzazione
import loss

# python3 main.py <isTraining> <isClassification> <epochsNumber> <filename> <filenameOldNetwork>

if len(sys.argv) < 6:
    print("Necessari argomenti")
    exit()

else:
    
    # lettura da riga di comando 
    isTraining = sys.argv[1] # se non è training si leggono i valori dei pesi dopo l'addestramento da file
    isClassification = sys.argv[2]
    numberEpochs = int(sys.argv[3]) # numero epoche per TR
    filename = sys.argv[4]
    filenameOldNetwork = sys.argv[5]

    if isTraining == '1':

        if isClassification == '1':

            # lettura TR
            training_set = read_data.read_forClassification(filename)
            inputs = np.array(training_set[0])
            targets = np.array(training_set[1])
            
            # layer_sizes => dimensioni degli hidden layers, seguite dalle dimensioni dell'output layer
            layer_sizes = [len(inputs[0]), 10, 1]
            # learning rate 
            learning_rate = 0.1
            # creazione e addestramento rete
            network_instance = nn.NN(layer_sizes, learning_rate, activation_functions.sigmoid , activation_functions.sigmoid, loss.binary_crossentropy , 1)
            ret = network_instance.run_training(inputs,targets,numberEpochs)
            loss_values = ret[0]
            outputs = ret[1] # output finali dopo tutte le epoche 
            # calcolo accuracy per ogni epoca ( e trasformo l'output in 0 e 1 )
            accuracy = []
            
            for out in outputs: 
                out = np.ravel(out)
                for i in range(len(out)):
                    
                    if out[i] >= 0.5:
                        out[i] = 1
                    else:
                        out[i] = 0
                print(out)
                accuracy.append(loss.percentClassification(targets,out))

            # Lista del numero di esempi
            epochs = list(range(1, len(loss_values) + 1))


            # GRAFICO LOSS 
            plt.plot(epochs, loss_values , marker='o', linestyle='-', color='r')
            plt.xlabel('Numero di epoche')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.show()

            # Lista numero di epoche 

            # GRAFICO ACCURACY
            plt.plot(epochs, accuracy , marker='o', linestyle='-', color='b')
            plt.xlabel('Numero di epoche')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.show()


        else: # regressione
            print("Regressione")
            exit() 



    else: # isTest 
        # leggo da file i valori dei pesi dopo l'addestramento
        try:
            # Per caricare l'istanza della rete da un file
            with open(filenameOldNetwork, 'rb') as file:
                network_instance = pickle.load(file)
        
        except FileNotFoundError:
            print("Il file non esiste.")
            exit()
        
        # esecuzione del test sul file di test





































        # if isClassification == '0':
            
        #     # NORMALIZZAZIONE DEI DATI
        #     training_set = np.array(training_set)
        #     training_set = training_set.astype(float)
        #     mean_values = np.mean(training_set, axis=0) #mean_values e std_dev_values servono dopo per il TS
        #     std_dev_values = np.std(training_set, axis=0)
        #     training_set = read_data.normalize_data(training_set, std_dev_values, mean_values)

        #     # divido in id , input , target dei TR e salvo nelle mie strutture dati
        #     # ids = [] # lista degli id del TR => non serve a niente

        #     inputs = [] # lista degli array di input 
        #     targets = [] #lista degli array target
        #     for riga in training_set:
        #         inputs.append(list(map(float, riga[0:-3])))
        #         targets.append(list(map(float, riga[-3:])))

        #     # se leggo i valori della rete da file
        #     if network_is_new == '0':
        #         try:
        #             # Per caricare l'istanza della rete da un file
        #             with open(filenameOldNetwork, 'rb') as file:
        #                 network_instance = pickle.load(file)
            
        #         except FileNotFoundError:
        #             print("Il file non esiste.")
        #             exit()
        #     # rete nuova
        #     else:
        #         # da rivedere
        #         network_instance = nn.NN(len(inputs[0]), len(targets[0]), 15, 1, 0.1, activation_functions.relu, activation_functions.sigmoid,1)
            
        #     meanerrors = network_instance.run_training(inputs,targets,numberEpochs)
        #     # print(meanerrors)
        #     # Salva l'istanza della rete su un file => NON SO SE FUNZIONA
        #     with open('rete.pkl', 'wb') as file:
        #         pickle.dump(network_instance, file)


        #     # STAMPO GRAFICO DELLA MSE 
        #     # Lista del numero di epoche
        #     epochs = list(range(1, len(meanerrors) + 1))
        #     # Creazione del grafico
        #     plt.plot(epochs, meanerrors, marker='o', linestyle='-', color='b')
        #     plt.title('Mean Squared Error (MSE) rispetto al numero di epoche')
        #     plt.xlabel('Numero di epoche')
        #     plt.ylabel('MSE')
        #     plt.grid(True)
        #     plt.show()

        # else: # isClassification
           
    


