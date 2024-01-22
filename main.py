import read_data
import activation_functions
import nn 
import numpy as np
import pickle # per salvare la rete
import sys
import matplotlib.pyplot as plt # visualizzazione
import loss

# python3 main.py <isTraining> <isClassification> <epochsNumber> <filename> <filenameToSave>

# filenameToSave è il nome del file su cui salvare la rete dopo l'addestramento o da cui leggere i valori della rete in caso di test  
if len(sys.argv) < 6:
    print("Necessari argomenti")
    exit()

else:
    
    # lettura da riga di comando 
    isTraining = sys.argv[1] # se non è training si leggono i valori dei pesi dopo l'addestramento da file
    isClassification = sys.argv[2] # per adesso non viene usato ==> aggiungere poi per la regressione 
    numberEpochs = int(sys.argv[3]) # numero epoche per TR
    filename = sys.argv[4]
    filenameToSave = sys.argv[5]


    # rete nuova da addestrare 
    if isTraining == "TR":
        
        # classificazione 
        if isClassification == '1':

            # lettura TR
            training_set = read_data.read_forClassification(filename)
            inputs = np.array(training_set[0])
            targets = np.array(training_set[1])
            
            # layer_sizes => dimensioni degli hidden layers, seguite dalle dimensioni dell'output layer
            layer_sizes = [len(inputs[0]), 5 , 5 , 1]
            # learning rate 
            learning_rate = 0.1
            # creazione e addestramento rete
            network_instance = nn.NN(layer_sizes, learning_rate, activation_functions.sigmoid , activation_functions.sigmoid, loss.binary_crossentropy , filenameToSave)
            ret = network_instance.run_training(inputs,targets,numberEpochs)
            loss_values = ret[0]
            outputs = ret[1] # output finali dopo tutte le epoche 

            # calcolo accuracy per ogni epoca 
            accuracy = []
            # trasformo ogni output in {0,1}
            for out in outputs: 
                out = np.ravel(out)
                for i in range(len(out)):
                    
                    if out[i] >= 0.5:
                        out[i] = 1
                    else:
                        out[i] = 0
                accuracy.append(loss.percentClassification(targets,out))



            # Lista del numero di epoche
            epochs = list(range(1, len(loss_values) + 1))


            # GRAFICO LOSS 
            plt.plot(epochs, loss_values , marker='o', linestyle='-', color='r')
            plt.xlabel('Numero di epoche')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.show()

            # GRAFICO ACCURACY
            plt.plot(epochs, accuracy , marker='o', linestyle='-', color='b')
            plt.xlabel('Numero di epoche')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.show()


        else: # regressione
            print("Regressione")
            exit() 



    # isTest 
    elif isTraining == "TS": 

        # leggo da file i valori dei pesi dopo l'addestramento
        try:
            # Per caricare l'istanza della rete da un file
            with open(filenameToSave, 'rb') as file:
                network_instance = pickle.load(file)
        
        except FileNotFoundError:
            print("Il file non esiste.")
            exit()
        
        # lettura TS 
        test_set = read_data.read_forClassification(filename)
        inputs = np.array(test_set[0])
        targets = np.array(test_set[1])

        # esecuzione del test sul file di test
        outputs = network_instance.run_test(inputs)
        for i in range(len(outputs)):
            if outputs[i] >= 0.5:
                outputs[i] = 1
            else:
                outputs[i] = 0
        accuracy = loss.percentClassification(targets,outputs)
        print("Accuracy: ", accuracy)


    else: 
        print("Specificare da riga di comando se si vuole eseguire un addestramento (TR) o un test (TS)")
        exit()





























