import numpy as np
import sys



def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def mean_squared_error_derivative(y_true, y_pred):
    return 2*(y_pred - y_true)

#########################################################################################################
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mae_derivative(y_true, y_pred):
    n = len(y_true)
    return np.sign(y_pred - y_true) / n

#########################################################################################################
def rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse_value = np.sqrt(mse)
    return rmse_value

def rmse_derivative(y_true, y_pred):
    n = len(y_true)
    error = y_pred - y_true
    mse_derivative = 2 * np.mean(error)
    rmse_derivative = 0.5 * np.sqrt(np.maximum(mse_derivative, 1e-10))
    return rmse_derivative

#########################################################################################################

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    huber_condition = np.abs(error) < delta
    quadratic_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(huber_condition, quadratic_loss, linear_loss))



def huber_derivative(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    n = len(y_true)
    
    huber_condition = np.abs(error) < delta
    derivative_quadratic = error / n
    derivative_linear = delta * np.sign(error) / n

    return np.where(huber_condition, derivative_quadratic, derivative_linear)

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


def binary_crossentropy_derivative(y_true, y_pred):
    # Calcola la derivata della Binary Crossentropy rispetto a y_pred
    return - (y_true / y_pred - (1 - y_true) / (1 - y_pred))

#########################################################################################################

def derivative(function):
    if function == binary_crossentropy:
        return binary_crossentropy_derivative
    if function == mean_squared_error:
        return mean_squared_error_derivative
    if function == mae:
        return mae_derivative
    if function == huber_loss:
        return huber_derivative
    if function == rmse:
        return rmse_derivative
    else:
        print("loss:derivative:ERROR")
        exit()

#########################################################################################################