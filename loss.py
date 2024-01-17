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