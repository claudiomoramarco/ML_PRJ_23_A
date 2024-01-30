import loss
import numpy as np
import pickle # per salvare la rete
import sys
import activation_functions

class NN:

    # layer sizes passato dall'esterno
    def __init__(self, layer_sizes, learningRate, momentum, activationFunctionForHidden, activationFunctionForOutput, lossFunction, filenameToSave, l2, regularization_coefficient):
        
        self.activationFunctionForHidden = activationFunctionForHidden
        self.activationFunctionForOutput = activationFunctionForOutput
        self.lossFunction = lossFunction
        self.layer_sizes = layer_sizes # contiene la lista del numero di neuroni di ogni layer
        self.filenameToSave = filenameToSave
        self.learningRate = learningRate
        self.momentum = momentum
        self.l2 = l2
        self.regularization_coefficient = regularization_coefficient
        self.initializeWeigths()
        self.initialize_velocity()

#########################################################################################################
        
    
    # Inizializza pesi e bias in modo casuale
    def initializeWeigths(self):
        np.random.seed(42)
        self.weights_hiddens = []
        self.bias_hiddens = []

        # per gli hidden  
        for i in range(len(self.layer_sizes)-2):
            self.weights_hiddens.append(np.random.uniform(-0.1, 0.1,(self.layer_sizes[i] , self.layer_sizes[i+1]) ))
            self.bias_hiddens.append(np.full(self.layer_sizes[i+1], 0))

        # per l'output layer 
        self.weights_output = np.random.uniform(-0.1, 0.1,(self.layer_sizes[-2], self.layer_sizes[-1]) )
        self.bias_output = np.full(self.layer_sizes[-1], 0)
   
#########################################################################################################
    
    def initialize_velocity(self):
        self.velocity_weights = []
        self.velocity_biases = []
        for i in range(len(self.weights_hiddens)):
            self.velocity_weights.append(np.zeros_like(self.weights_hiddens[i]))
            self.velocity_biases.append(self.bias_hiddens[i])
        self.velocity_weights.append(np.zeros_like(self.weights_output))
        self.velocity_biases.append(self.bias_output)

#########################################################################################################
    def update_weights(self, weights, gradients, velocities):
        
        velocities = self.momentum * velocities + (1 - self.momentum) * gradients
    
        # regolarizzazione L2
        if self.l2:
            weights = weights - self.learningRate * (velocities + self.regularization_coefficient * weights)
        else: 
            weights = weights - self.learningRate * velocities

        return (weights, velocities)
    
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
        # final_output è l'output dell'output layer , hidden_outputs è l'array degli output di ogni hidden layer
        return (hidden_outputs,final_output) 
    

#########################################################################################################
    
    # final_output è l'output del pattern considerato
    # hidden_outputs[i] è l'output dell'hidden layer i-esimo del pattern considerato
    def backpropagation(self, d_loss, final_output, hidden_outputs , data, target):
        
        # gradiente output layer 
        delta_out = (activation_functions.derivative(self.activationFunctionForOutput)(np.dot(hidden_outputs[-1], self.weights_output))) * d_loss
        grad_out_weigths = np.outer(hidden_outputs[-1], delta_out) 
        grad_out_bias = delta_out

        # gradiente hidden layers
        grad_hid_weigths = [] 
        delta_hiddens = []
        grad_hid_bias = []

        if len(self.weights_hiddens) == 1: # se c'è un solo hidden layer 
            delta_hiddens.append(np.dot(delta_out, np.transpose(self.weights_output)) * 
                                     activation_functions.derivative(self.activationFunctionForHidden)(np.dot(data,self.weights_hiddens[0])))
            grad_hid_weigths.append(np.outer(data, delta_hiddens[-1]))
            grad_hid_bias.append(delta_hiddens[-1])

        else:
            for i in range(len(self.weights_hiddens)-1, -1, -1):
                if i == len(self.weights_hiddens)-1 :
                    delta_hiddens.append(np.dot(delta_out, np.transpose(self.weights_output)) * 
                                        activation_functions.derivative(self.activationFunctionForHidden)(np.dot(hidden_outputs[i-1],self.weights_hiddens[i])))
                    grad_hid_weigths.append(np.outer(hidden_outputs[i-1], delta_hiddens[-1]))
                elif i == 0:
                    delta_hiddens.append(np.dot(delta_hiddens[-1], np.transpose(self.weights_hiddens[i+1])) * 
                                        activation_functions.derivative(self.activationFunctionForHidden)(np.dot(data,self.weights_hiddens[i])))
                    grad_hid_weigths.append(np.outer(data, delta_hiddens[-1]))
                else:
            
                    delta_hiddens.append(np.dot(delta_hiddens[-1], np.transpose(self.weights_hiddens[i+1]))*
                                        activation_functions.derivative(self.activationFunctionForHidden)(np.dot(hidden_outputs[i-1],self.weights_hiddens[i])))
                    grad_hid_weigths.append(np.outer(hidden_outputs[i-1], delta_hiddens[-1]))

                grad_hid_bias.append(delta_hiddens[-1])
        
        # inverto l'ordine delle matrici perché sono stati riempiti al contrario
        grad_hid_weigths = grad_hid_weigths[::-1]      
        grad_hid_bias = grad_hid_bias[::-1]
        return(grad_hid_weigths,grad_out_weigths, grad_hid_bias, grad_out_bias) # con lo stesso criterio di forwardpropagation
    

#########################################################################################################
    
    def run_training(self, tr_data, tr_targets, numberEpochs, stop, batch_size, test_data, test_targets):
        
        loss_tot_TR = []
        outputs_tot_TR = []
        # loss_tot_TS = []
        # outputs_tot_TS = []

        # Addestramento del modello
        for epoch in range(numberEpochs):

            output_epoch_TR = []
            output_epoch_TS = []
            # somma delle loss dell'epoca
            loss_sum_TR = 0  
            # training set
            for i in range(len(tr_data)): 

                # batch
                if i%batch_size == 0: 
                    # dichiara vuota l'array delle derivate della loss del batch 
                    d_loss_batch = []
                # Forward propagation
                ret = self.forwardpropagation(tr_data[i])
                hidden_outputs = ret[0] # array degli output per ogni hidden  layer 
                final_output = ret[1] # output dell'output layer
                output_epoch_TR.append(final_output)
                # calcolo della loss dell'esempio corrente
                loss_value = self.lossFunction(tr_targets[i], final_output)                 
                # Calcolo della derivata della funzione di loss rispetto a y_pred 
                d_loss = loss.derivative(self.lossFunction)(tr_targets[i], final_output)
  
                # salvo tra quelle del batch
                d_loss_batch.append(d_loss)
                
                # ogni batch_size patterns oppure se è l'ultimo pattern 
                if i%batch_size == batch_size-1 or i == len(tr_data)-1:
                    

                    # media delle derivate nel batch 
                    d_loss_avg = np.mean(d_loss_batch, axis=0)

                    # Backpropagation
                    ret = self.backpropagation(d_loss_avg, final_output, hidden_outputs, tr_data[i], tr_targets[i])
                    
                    grad_hiddens = ret[0]
                    grad_output = ret[1]
                    grad_hid_bias = ret[2]
                    grad_out_bias = ret[3]
                    
                    # aggiornamento dei pesi 
                 
                    # per gli hidden layers  
                    for j in range(len(self.weights_hiddens)):
                    
                        self.weights_hiddens[j], self.velocity_weights[j] = self.update_weights(self.weights_hiddens[j], grad_hiddens[j], self.velocity_weights[j])
                        # aggiornamento bias 
                        self.bias_hiddens[j], self.velocity_biases[j] = self.update_weights(self.bias_hiddens[j], grad_hid_bias[j], self.velocity_biases[j])
                    
                    # per l'output layer
                    self.weights_output, self.velocity_weights[-1] = self.update_weights(self.weights_output, grad_output, self.velocity_weights[-1])
                    self.bias_output, self.velocity_biases[-1] = self.update_weights(self.bias_output, grad_out_bias, self.velocity_biases[-1])
                
                # somma della loss per ogni esempio per poi farne la media dell'epoca
                loss_sum_TR += loss_value 

            # test set 
            loss_sum_TS = 0 # somma della loss di ogni esempio dell'epoca corrente 
            for i in range(len(test_data)):
                # Forward propagation
                ret = self.forwardpropagation(test_data[i])
                final_output = ret[1]
                output_epoch_TS.append(final_output)
                # calcolo della loss 
                loss_value = self.lossFunction(test_targets[i], final_output)
                loss_sum_TS += loss_value


            # salva la loss media per ogni epoca
            loss_tot_TR.append(loss_sum_TR/len(tr_data))
            # loss_tot_TS.append(loss_sum_TS/len(test_data))

            # salva la lista degli output per ogni epoca in modo da poter calcolare l'accuracy
            outputs_tot_TR.append(output_epoch_TR) 

            # ogni 20 epoche stampa la loss 
            if stop == 0: # ovvero è stato chiamato da main.py
                if len(outputs_tot_TR)%20 == 0:
                    print(loss_tot_TR[-1])


        # salva su file alla fine dell'addestramento 
        self.save_to_file()

        return (loss_tot_TR, outputs_tot_TR)



#########################################################################################################

    def run_test(self, test_data):
        
        output = []
        for i in range(len(test_data)):
            # Forward propagation
            ret = self.forwardpropagation(test_data[i])
            final_output = ret[1]
            output.append(final_output)
        
        return output
    
#########################################################################################################