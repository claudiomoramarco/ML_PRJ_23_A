# import nn
import read_data
import numpy as np 
import pickle # per salvare la rete
import loss


def classificationTesting(filename,filenameToSave):
        
    # leggo da file i valori della NN
    try:
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


#########################################################################################################
    

def regressionTesting(filename,filenameToSave):
      # leggo da file i valori della NN
    try:
        with open(filenameToSave, 'rb') as file:
            network_instance = pickle.load(file)
    
    except FileNotFoundError:
        print("Il file non esiste.")
        exit()

    # lettura TS 
    test_set = read_data.readForRegression(filename)
    inputs = np.array(test_set[0])
    targets = np.array(test_set[1])

    # esecuzione del test sul file di test
    outputs = network_instance.run_test(inputs)

    print(outputs[-1][-1])
    loss_value = loss.mean_squared_error(targets,outputs)
    print("Loss sul TS: ", loss_value)