import read_data
import numpy as np 
import activation_functions
import loss
import matplotlib.pyplot as plt # visualizzazione
import nn


# l2 è un booleano che è true se si vuole usare la regolarizzazione L2 
def classificationTraining(filename, numberEpochs, filenameToSave, layer_sizes, learning_rate, momentum, batch_size, l2, regularization_coefficient, filenameToTest):
    # lettura TR
    training_set = read_data.read_forClassification(filename)
    inputs_TR = np.array(training_set[0])
    targets_TR = np.array(training_set[1])
    # lettura TS (ma poi non viene usato)
    test_set = read_data.read_forClassification(filenameToTest)
    inputs_TS = np.array(test_set[0])
    targets_TS = np.array(test_set[1])
    # aggiungo l'input layer 
    layer_sizes = np.insert(layer_sizes, 0 , len(inputs_TR[0]))
    # creazione e addestramento rete
    network_instance = nn.NN(layer_sizes, learning_rate, momentum, activation_functions.relu , activation_functions.sigmoid, loss.binary_crossentropy , filenameToSave, l2, regularization_coefficient)
    ret = network_instance.run_training(inputs_TR,targets_TR,numberEpochs, -1, batch_size,inputs_TS, targets_TS) 
    loss_values_TR = ret[0] # valori della loss per ogni epoca 
    outputs = ret[1] # output finali dopo tutte le epoche 
    # loss_values_TS = ret[2]
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
        accuracy.append(loss.percentClassification(targets_TR,out))
    # Lista del numero di epoche
    epochs = list(range(1, len(loss_values_TR) + 1))
    plt.plot(epochs, loss_values_TR, marker='o', linestyle='-', color='b', label='Loss TR', linewidth=2, markersize=2)
    # plt.plot(epochs, loss_values_TS, marker='o', linestyle='-', color='g', label='Loss TS', linewidth=2, markersize=2)
    plt.xlabel('Numero di epoche')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    # GRAFICO ACCURACY
    plt.plot(epochs, accuracy , marker='o', linestyle='-', color='b', linewidth=1, markersize=1)
    plt.xlabel('Numero di epoche')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()




#########################################################################################################

# stop è la soglia del valore della loss per fermarsi
def regressionTraining(filename, numberEpochs, filenameToSave, layer_sizes, learning_rate, momentum, stop, batch_size, l2, regularization_coefficient):
    # lettura TR
    training_set = read_data.readForRegression(filename)
    inputs = np.array(training_set[0])
    targets = np.array(training_set[1])
    # Aggiunta input layer e output layer 
    layer_sizes = np.insert(layer_sizes, 0 , len(inputs[0]))
    layer_sizes = np.insert(layer_sizes, len(layer_sizes), len(targets[0]))
    # creazione e addestramento rete
    network_instance = nn.NN(layer_sizes, learning_rate, momentum, activation_functions.relu , activation_functions.linear, loss.mean_squared_error , filenameToSave, l2, regularization_coefficient)
    ret = network_instance.run_training(inputs, targets, numberEpochs, stop, batch_size)
    loss_values = ret[0]
    outputs = ret[1] # output finali dopo tutte le epoche 
    # stampo ultimo valore calcolato
    print("ultimo valore:")
    print(outputs[-1])    
    print(targets)
    # Lista del numero di epoche
    epochs = list(range(1, len(loss_values) + 1))
    # GRAFICO LOSS 
    plt.plot(epochs, loss_values , marker='o', linestyle='-', color='r')
    plt.xlabel('Numero di epoche')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


#########################################################################################################

def classificationTRTS(filenameTR, filenameTS, numberEpochs, filenameToSave, layer_sizes, learning_rate, momentum, batch_size, l2, regularization_coefficient):
    
    # lettura TR
    training_set = read_data.read_forClassification(filenameTR)
    inputs_TR = np.array(training_set[0])
    targets_TR = np.array(training_set[1])
    # lettura TS 
    test_set = read_data.read_forClassification(filenameTS)
    inputs_TS = np.array(test_set[0])
    targets_TS = np.array(test_set[1])
    # aggiungo l'input layer 
    layer_sizes = np.insert(layer_sizes, 0 , len(inputs_TR[0]))
    # creazione e addestramento rete
    network_instance = nn.NN(layer_sizes, learning_rate, momentum, activation_functions.relu , activation_functions.sigmoid, loss.binary_crossentropy , filenameToSave, l2, regularization_coefficient)
    ret = network_instance.run_training(inputs_TR,targets_TR,numberEpochs, -1, batch_size,inputs_TS, targets_TS) 
    loss_values_TR = ret[0] # valori della loss per ogni epoca 
    outputs = ret[1] # output finali dopo tutte le epoche 
    loss_values_TS = ret[2]
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
        accuracy.append(loss.percentClassification(targets_TR,out))
    # Lista del numero di epoche
    epochs = list(range(1, len(loss_values_TR) + 1))
    plt.plot(epochs, loss_values_TR, marker='o', linestyle='-', color='b', label='Loss TR', linewidth=2, markersize=2)
    plt.plot(epochs, loss_values_TS, marker='o', linestyle='-', color='g', label='Loss TS', linewidth=2, markersize=2)
    plt.xlabel('Numero di epoche')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    # GRAFICO ACCURACY
    plt.plot(epochs, accuracy , marker='o', linestyle='-', color='b', linewidth=1, markersize=1)
    plt.xlabel('Numero di epoche')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
