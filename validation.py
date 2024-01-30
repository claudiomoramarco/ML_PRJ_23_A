import numpy as np 
import read_data
import sys
import nn
import activation_functions
import loss


# <validation.py> <data_filename> <parameters_filename> <isRegression> <k>

def kFoldCrossValidation(k, isRegression, data_filename, parameters_filename):  #filenameToSave, layer_sizes, learning_rate, momentum, batch_size, l2, regularization_coefficient):

    # lettura TR
    if isRegression:
        training_set = read_data.readForRegression(data_filename)
    else:
        training_set = read_data.read_forClassification(data_filename)

    inputs = np.array(training_set[0])
    targets = np.array(training_set[1])

    # creazione fold
    input_folds, target_folds = createKFolds(k, inputs, targets)

    # lettura file parametri
    configs = readParameters(parameters_filename)

    computedLossConfigurations =[]

    # per ogni configurazione 
    for i in range(len(configs)):
        print("Configurazione :", i)

        if isRegression:
            # Aggiunta input layer e output layer 
            layer_sizes = configs[i]['hidden_layer_sizes']
            layer_sizes = np.insert(layer_sizes, 0 , len(inputs[0]))
            layer_sizes = np.insert(layer_sizes, len(layer_sizes), len(targets[0]))
            # creazione rete 
            network_instance = nn.NN(layer_sizes, configs[i]['learning_rate'], configs[i]['momentum'], activation_functions.relu , activation_functions.linear , loss.mean_squared_error, 'val', configs[i]['l2'], configs[i]['regularization_coefficient'])
        else:
            layer_sizes = configs[i]['hidden_layer_sizes']
            # aggiungo l'input layer 
            layer_sizes = np.insert(layer_sizes, 0 , len(inputs[0]))
            # creazione rete
            network_instance = nn.NN(layer_sizes, configs[i]['learning_rate'], configs[i]['momentum'], activation_functions.sigmoid , activation_functions.sigmoid , loss.binary_crossentropy, 'val', configs[i]['l2'], configs[i]['regularization_coefficient'])
          
        loss_avg_config = 0

        # addestramento per ogni fold
        for j in range(k):

            dataTR = None  
            targetsTR = None
            # creazione TR
            for l in range(k):
                if j != l:
                    if dataTR is None:
                        dataTR = input_folds[l]
                        targetsTR = target_folds[l]
                    else:
                        dataTR = np.vstack((dataTR, input_folds[l]))
                        targetsTR = np.vstack((targetsTR, target_folds[l]))
            if not isRegression:
                targetsTR = np.ravel(targetsTR)
            
            # addestramento
            network_instance.run_training(dataTR, targetsTR, 20 , -1, configs[i]['batch_size'], [], [])
            # test su input_folds[j]
            ret = network_instance.run_test(input_folds[j])
            loss_avg_config += loss.mean_squared_error(target_folds[j], ret)
        
        # loss media dei fold
        computedLossConfigurations.append(loss_avg_config/k)
        print("Loss media: " , computedLossConfigurations[-1])
    

    # cerco la miglior configurazione   
    min = np.argmin(computedLossConfigurations)
    # stampa i risultati su file 
    write_results(configs, computedLossConfigurations)

    print("La configurazione migliore e' ", min, " con un valore di loss di ", computedLossConfigurations[min])



#########################################################################################################

def createKFolds(k, inputs, targets):
    if k < 1:
        print("Errore nel numero di fold, deve essere maggiore di 1")
        exit()

    if (len(inputs)%k != 0):
        print("Errore nel numero di fold, deve essere divisibile")
        exit()
    
    # numero di pattern per ogni fold
    N = len(inputs)/k

    input_folds = []
    taregt_folds = []

    for i in range(k):
        input_folds.append(inputs[int(i*N):int((i+1)*N)])
        taregt_folds.append(targets[int(i*N):int((i+1)*N)])

                      
    return np.array(input_folds), np.array(taregt_folds)


#########################################################################################################

def readParameters(filename):
 
 with open(filename, 'r') as file:
        
        configurazioni = []

        # Ignora la prima riga
        next(file)
        
        for riga in file:
            
            # suddividi i parametri utilizzando il punto e virgola
            parametri = riga.split(';')
            # Estrai i valori necessari dai parametri
            hidden_layer_sizes = np.array(eval(parametri[0]))
            learning_rate = float(parametri[1])
            momentum = float(parametri[2])
            l2 = bool(int(parametri[3]))  # Converto il valore a booleano
            regularization_coefficient = float(parametri[4])
            batch_size = int(parametri[5])
            # Aggiungi i parametri alla lista
            configurazioni.append({
                'hidden_layer_sizes': hidden_layer_sizes,
                'learning_rate': learning_rate,
                'momentum': momentum,
                'l2': l2,
                'regularization_coefficient': regularization_coefficient,
                'batch_size': batch_size
            })
            
        return configurazioni
        
#########################################################################################################

def write_results(configs, computedLossConfigurations):
    with open('avg_loss_validation.txt', 'w') as file:
        for indice, (config, loss) in enumerate(zip(configs, computedLossConfigurations)):
            line = f"{indice}: {config} : Loss {loss}\n"
            file.write(line)
            
#########################################################################################################

data_filename = sys.argv[1]
parameters_filename = sys.argv[2]
if sys.argv[3] == '1':
    isRegression = 1
else:
    isRegression = 0
k = int(sys.argv[4])
kFoldCrossValidation(k, isRegression, data_filename, parameters_filename)

