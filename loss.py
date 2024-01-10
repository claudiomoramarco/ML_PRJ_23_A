import numpy as np

#calcola la MSE dopo un'epoca
def mse(tr_targets, outputs):
    if len(outputs) != len(tr_targets):
        print("loss:mse:ERROR")
        exit()
    
    # Conversione delle liste in array NumPy
    outputs = np.array(outputs)
    tr_targets = np.array(tr_targets)
    
    # print(outputs[10])
    # print(tr_targets[10])

    mean_squared = np.mean(np.square(outputs - tr_targets))
    return mean_squared