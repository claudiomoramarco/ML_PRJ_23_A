import read_data
import numpy as np 
import activation_functions
import loss
import matplotlib.pyplot as plt # visualizzazione
import nn


def classificationTraining(filename, numberEpochs, filenameToSave, layer_sizes, learning_rate):
    
    # lettura TR
    training_set = read_data.read_forClassification(filename)
    inputs = np.array(training_set[0])
    targets = np.array(training_set[1])
    
    # aggiungo l'input layer 
    layer_sizes = np.insert(layer_sizes, 0 , len(inputs[0]))
    
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




    #########################################################################################################

def regressionTraining(filename, numberEpochs, filenameToSave, layer_sizes, learning_rate):

    # lettura TR
    training_set = read_data.readForRegression(filename)
    inputs = np.array(training_set[0])
    targets = np.array(training_set[1])

    # Aggiunta input layer e output layer 
    layer_sizes = np.insert(layer_sizes, 0 , len(inputs[0]))
    layer_sizes = np.insert(layer_sizes, len(layer_sizes), len(targets[0]))

    # creazione e addestramento rete
    network_instance = nn.NN(layer_sizes, learning_rate, activation_functions.relu , activation_functions.relu, loss.mean_squared_error , filenameToSave)
    ret = network_instance.run_training(inputs, targets, numberEpochs)
    
    print(numberEpochs)
    loss_values = ret[0]
    outputs = ret[1] # output finali dopo tutte le epoche 
    
    # Lista del numero di epoche
    epochs = list(range(1, len(loss_values) + 1))

    # GRAFICO LOSS 
    plt.plot(epochs, loss_values , marker='o', linestyle='-', color='r')
    plt.xlabel('Numero di epoche')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()