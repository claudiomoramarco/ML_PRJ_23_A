import numpy as np

#calcola la MSE dopo un'epoca
def mse(tr_targets, outputs):
    if len(outputs) != len(tr_targets):
        print("loss:mse:ERROR")
        exit()
    
    # Conversione delle liste in array NumPy
    outputs = np.array(outputs)
    tr_targets = np.array(tr_targets)
    
    mean_squared = np.mean(np.square(outputs - tr_targets))
    return mean_squared


#########################################################################################################


# tr_targets e outputs sono due liste di booleani
def percentClassification(tr_targets, outputs):
    
    if len(tr_targets) != len(outputs):
        print("loss:percentClassification:ERROR")
        exit()
     
    correct = 0
    for i in range(len(tr_targets)):
        if tr_targets[i] == outputs[i]:
            correct+=1
    return (correct/len(tr_targets))*100

#########################################################################################################

def binary_crossentropy(y_true, y_pred):
    y_pred = np.ravel(y_pred)
    # Calcola la Binary Crossentropy per un singolo esempio
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

#########################################################################################################

def binary_crossentropy_derivative(y_true, y_pred):
    # Calcola la derivata della Binary Crossentropy rispetto a y_pred
    return - (y_true / y_pred - (1 - y_true) / (1 - y_pred))

#########################################################################################################

def update_weights(weights, gradients, learning_rate):
    # Aggiorna i pesi usando la discesa del gradiente
    return weights - learning_rate * gradients

#########################################################################################################

def derivative(function):
    if function == binary_crossentropy:
        return binary_crossentropy_derivative
