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
        self.weights_hiddens = []
        self.bias_hiddens = []

        # per gli hidden  
        for i in range(len(self.layer_sizes)-2):
            self.weights_hiddens.append(np.random.normal(0, 1,  (self.layer_sizes[i] , self.layer_sizes[i+1])))             
            self.bias_hiddens.append(np.zeros((1, self.layer_sizes[i+1])))

        # per l'output layer 
        self.weights_output = np.random.normal(0, 1, (self.layer_sizes[-2], self.layer_sizes[-1]))
        self.bias_output = np.zeros((1, self.layer_sizes[-1]))


#########################################################################################################
    
    def update_weights(self, weights, gradients, learning_rate):
        # print("pesi prima")
        # print(weights)
        # Aggiorna i pesi usando la discesa del gradiente
        return weights - learning_rate * gradients

#########################################################################################################
    
    def save_to_file(self):
        with open(self.filenameToSave, 'wb') as file:
            pickle.dump(self, file)
    
#########################################################################################################
            
    # data è l'iesimo pattern
    def forwardpropagation(self, data):

        hidden_outputs = []
        for i in range(len(self.weights_hiddens)): # per gli hidden
            if i == 0:
                hidden_outputs.append(np.ravel(self.activationFunctionForHidden(np.dot(data, self.weights_hiddens[i]) + self.bias_hiddens[i])))
            else: 
                hidden_outputs.append(np.ravel(self.activationFunctionForHidden(np.dot(hidden_outputs[i-1], self.weights_hiddens[i]) + self.bias_hiddens[i])))
        
        final_output = np.ravel(self.activationFunctionForOutput(np.dot(hidden_outputs[-1], self.weights_output) + self.bias_output))

        return (hidden_outputs,final_output) # final_output è l'output dell'output layer , hidden_outputs è l'array degli output di ogni hidden layer

#########################################################################################################
    
    # final_output è l'output del pattern considerato
    # hidden_outputs[i] è l'output dell'hidden layer i-esimo del pattern considerato
    # d_loss la derivata della loss rispetto all'output del pattern considerato
    def backpropagation(self, d_loss, final_output, hidden_outputs):
        d_output = d_loss * final_output * (1 - final_output) # per l'output layer
        d_hiddens = np.zeros_like(hidden_outputs)
        for i in range(len(self.weights_hiddens) - 1, -1, -1): #per gli hidden
            if i == len(self.weights_hiddens)-1 : 
                d_hiddens[i] = np.dot(d_output, np.transpose(self.weights_output)) * hidden_outputs[i] * (1 - hidden_outputs[i])
            else: 
                d_hiddens[i] = np.dot(d_hiddens[i+1], np.transpose(self.weights_hiddens[i+1])) * hidden_outputs[i] * (1 - hidden_outputs[i])
        
        return(d_hiddens,d_output) # con lo stesso criterio di forwardpropagation
    

#########################################################################################################
    
    def run_training(self, tr_data, tr_targets, numberEpochs):
        
        loss_tot = []
        outputs_tot = []


        # Addestramento del modello
        for epoch in range(numberEpochs):
            
            output_epoch = []

            for i in range(len(tr_data)):

                # Forward propagation
                ret = self.forwardpropagation(tr_data[i])
                hidden_outputs = ret[0] # array degli output per ogni hidden  layer 
                final_output = ret[1] # output dell'output layer
                output_epoch.append(final_output)

                # Calcolo della Loss
                loss_value = self.lossFunction(tr_targets[i], final_output)

                loss_sum = 0

                # Calcolo della derivata della Binary Crossentropy rispetto a y_pred
                d_loss = loss.derivative(self.lossFunction)(tr_targets[i], final_output)

                # Backpropagation
                ret = self.backpropagation(d_loss, final_output, hidden_outputs)
                d_hiddens = ret[0]
                d_output = ret[1]
                
        
                #  Calcolo dei gradienti e aggiornamento dei pesi
                # per l'output layer 
                self.weights_output = self.update_weights(self.weights_output, np.outer(hidden_outputs[-1], d_output), self.learningRate)
                self.bias_output = self.update_weights(self.bias_output, d_output, self.learningRate)
                

                # per gli altri 
                for j in range(len(self.weights_hiddens)):
 
                    if j == 0:
                        self.weights_hiddens[j] = self.update_weights(self.weights_hiddens[j], np.outer(tr_data[i], d_hiddens[j]), self.learningRate)
                    else:
                        self.weights_hiddens[j] = self.update_weights(self.weights_hiddens[j], np.outer(hidden_outputs[j-1], d_hiddens[j]), self.learningRate)
            
                    # aggiornamento bias 
                    self.bias_hiddens[j] = self.update_weights(self.bias_hiddens[j], np.sum(d_hiddens[j], axis=0, keepdims=True), self.learningRate)
            
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
            ret = self.forwardpropagation(test_data[i])
            final_output = ret[1]
            output.append(final_output)
        
        return output