import numpy as np
import sys
import training
import testing

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
            
            # layer_sizes => dimensioni degli hidden layers, seguite dalle dimensioni dell'output layer
            layer_sizes = [10 , 1]
            # learning rate 
            learning_rate = 0.1
            momentum = 0.1
            batch_size = 1 # per i monk la metto a 1
            training.classificationTraining(filename, numberEpochs, filenameToSave,layer_sizes, learning_rate, momentum, batch_size)


        else: # regressione
            # layer_sizes => dimensioni degli hidden layers 
            hidden_layer_sizes = [15,15,20]
            learning_rate = 0.001
            momentum = 0.1
            stop = 0.32
            batch_size = 5
            training.regressionTraining(filename, numberEpochs , filenameToSave, hidden_layer_sizes, learning_rate, momentum, stop, batch_size)


    # isTest 
    elif isTraining == "TS": 
        
        if isClassification == '1':
            testing.classificationTesting(filename, filenameToSave)
        
        # non ha senso fare il regression test su ML-CUP23-TS.csv perché non ci sono i target
        else: 
            testing.regressionTesting(filename, filenameToSave)

    else: 
        print("Specificare da riga di comando se si vuole eseguire un addestramento (TR) o un test (TS)")
        exit()



#########################################################################################################
        
