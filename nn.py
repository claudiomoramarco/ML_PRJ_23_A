import loss
import numpy as np
import pickle # per salvare la rete


class NN:

    # layer sizes passato dall'esterno
    def __init__(self, layer_sizes, learningRate, activationFunctionForHidden, activationFunctionForOutput, lossFunction, filenameToSave):
        
        self.learningRate = learningRate
        self.activationFunctionForHidden = activationFunctionForHidden
        self.activationFunctionForOutput = activationFunctionForOutput
        self.lossFunction = lossFunction
        self.layer_sizes = layer_sizes # contiene la lista del numero di neuroni di ogni layer
        self.weights = []
        self.biases = []
        self.filenameToSave = filenameToSave
        self.initializeWeigths()
        # self.isClassification = isClassification => servirà poi se voglio fare la regressione
    
#########################################################################################################
        
    # Inizializza pesi e bias in modo casuale
    def initializeWeigths(self):
        np.random.seed(42)
        self.weights_hidden = np.random.normal(0, 1,  (self.layer_sizes[0] , self.layer_sizes[1]))
        self.bias_hidden = np.zeros((1, self.layer_sizes[1]))
        self.weights_output = np.random.normal(0, 1, (self.layer_sizes[1], self.layer_sizes[2]))
        self.bias_output = np.zeros((1, self.layer_sizes[2]))

#########################################################################################################
    def update_weights(self, weights, gradients, learning_rate):
        # Aggiorna i pesi usando la discesa del gradiente
        return weights - learning_rate * gradients

#########################################################################################################
    
    def save_to_file(self):
        with open(self.filenameToSave, 'wb') as file:
            pickle.dump(self, file)
    
#########################################################################################################
            
    # data è l'iesimo pattern
    def forwadpropagation(self, data):
        hidden_output = np.ravel(self.activationFunctionForHidden(np.dot(data, self.weights_hidden) + self.bias_hidden))
        final_output = np.ravel(self.activationFunctionForOutput(np.dot(hidden_output, self.weights_output) + self.bias_output))
        return hidden_output,final_output

#########################################################################################################
    
    # final_output è l'output dell'iesimo pattern  
    # hidden_output è l'output dell'hidden layer dell'iesimo pattern 
    # d_loss la derivata della loss rispetto all'output dell'iesimo pattern 
    def backpropagation(self, d_loss, final_output, hidden_output):
        d_output = d_loss * final_output * (1 - final_output)
        d_hidden = np.dot(d_output, np.transpose(self.weights_output)) * hidden_output * (1 - hidden_output)
        return(d_hidden,d_output)
    

    def run_training(self, tr_data, tr_targets, numberEpochs):
        
        loss_tot = []
        outputs_tot = []


        # Addestramento del modello
        for epoch in range(numberEpochs):
            
            output_epoch = []

            for i in range(len(tr_data)):

                # Forward propagation
                ret = self.forwadpropagation(tr_data[i])
                hidden_output = ret[0]
                final_output = ret[1]
                output_epoch.append(final_output)

                # Calcolo della Loss
                loss_value = self.lossFunction(tr_targets[i], final_output)

                loss_sum = 0

                # Calcolo della derivata della Binary Crossentropy rispetto a y_pred
                d_loss = loss.derivative(self.lossFunction)(tr_targets[i], final_output)

                # Backpropagation
                ret = self.backpropagation(d_loss, final_output, hidden_output)
                d_hidden = ret[0]
                d_output = ret[1]

                #  Calcolo dei gradienti e aggiornamento dei pesi
                self.weights_output = self.update_weights(self.weights_output, np.outer(hidden_output, d_output), self.learningRate)
                self.bias_output = self.update_weights(self.bias_output, d_output, self.learningRate)
                self.weights_hidden = self.update_weights(self.weights_hidden, np.outer(tr_data[i], d_hidden), self.learningRate)
                self.bias_hidden = self.update_weights(self.bias_hidden, np.sum(d_hidden, axis=0, keepdims=True), self.learningRate)
                
                # somma della loss per ogni esempio per poi farne la media dell'epoca
                loss_sum += loss_value 

            # salva la loss media per ogni epoca
            loss_tot.append(loss_sum/len(tr_data)) 
            # salva la lista degli output per ogni epoca in modo da poter calcolare l'accuracy
            outputs_tot.append(output_epoch) 


        # salva su file alla fine dell'addestramento 
        self.save_to_file()

        return (loss_tot, outputs_tot)

#########################################################################################################

    def run_test(self, test_data):
        
        output = []
        for i in range(len(test_data)):
            # Forward propagation
            ret = self.forwadpropagation(test_data[i])
            final_output = ret[1]
            output.append(final_output)
        
        return output