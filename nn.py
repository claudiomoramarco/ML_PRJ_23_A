import activation_functions
import loss
import numpy as np

class NN:

    # layer sizes passato dall'esterno
    def __init__(self, layer_sizes, learningRate, activationFunctionForHidden, activationFunctionForOutput, lossFunction, isClassification):
        self.learningRate = learningRate
        # self.isClassification = isClassification
        # self.activationFunctionForHidden = activationFunctionForHidden
        # self.activationFunctionForOutput = activationFunctionForOutput
        # self.lossFunction = lossFunction
        self.layer_sizes = layer_sizes # contiene la lista del numero di neuroni di ogni layer
        self.weights = []
        self.biases = []
        # self.initialize_weights()

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def binary_crossentropy(self,y_true, y_pred):
        y_pred = np.ravel(y_pred)
        # Calcola la Binary Crossentropy per un singolo esempio
        return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def binary_crossentropy_derivative(self, y_true, y_pred):
        # Calcola la derivata della Binary Crossentropy rispetto a y_pred
        return - (y_true / y_pred - (1 - y_true) / (1 - y_pred))

    def update_weights(self, weights, gradients, learning_rate):
        # Aggiorna i pesi usando la discesa del gradiente
        return weights - learning_rate * gradients
    
    def run_training(self, tr_data, tr_targets, numberEpochs):

        
        loss_tot = []
        outputs_tot = []

        # Inizializza pesi e bias in modo casuale
        np.random.seed(42)
        weights_hidden = np.random.normal(0, 1,  (self.layer_sizes[0] , self.layer_sizes[1]))
        bias_hidden = np.zeros((1, self.layer_sizes[1]))
        weights_output = np.random.normal(0, 1, (self.layer_sizes[1], self.layer_sizes[2]))
        bias_output = np.zeros((1, self.layer_sizes[2]))
    

        # Addestramento del modello
        for epoch in range(numberEpochs):
            
            output_epoch = []

            for i in range(len(tr_data)):
                # Forward propagation
                hidden_output = np.ravel(self.sigmoid(np.dot(tr_data[i], weights_hidden) + bias_hidden))
                final_output = np.ravel(self.sigmoid(np.dot(hidden_output, weights_output) + bias_output))

                output_epoch.append(final_output)

                # Calcolo della Binary Crossentropy
                loss = self.binary_crossentropy(tr_targets[i], final_output)
                # loss_tot.append(loss)  # serve per il return
                loss_sum = 0

                # Calcolo della derivata della Binary Crossentropy rispetto a y_pred
                d_loss = self.binary_crossentropy_derivative(tr_targets[i], final_output)

                # Backpropagation
                d_output = d_loss * final_output * (1 - final_output)
                d_hidden = np.dot(d_output, np.transpose(weights_output)) * hidden_output * (1 - hidden_output)

                #  Calcolo dei gradienti e aggiornamento dei pesi
                weights_output = self.update_weights(weights_output, np.outer(hidden_output, d_output), self.learningRate)
                bias_output = self.update_weights(bias_output, d_output, self.learningRate)
                weights_hidden = self.update_weights(weights_hidden, np.outer(tr_data[i], d_hidden), self.learningRate)
                bias_hidden = self.update_weights(bias_hidden, np.sum(d_hidden, axis=0, keepdims=True), self.learningRate)
                
                loss_sum += loss

            # Stampa la loss ogni epoca
            print("Loss: ", loss_sum/len(tr_data))
            loss_tot.append(loss_sum/len(tr_data))
            outputs_tot.append(output_epoch)


        return (loss_tot, outputs_tot)
                # # Forward propagation
                # hidden_output = np.ravel(self.sigmoid(np.dot(tr_data[i], weights_hidden)+bias_hidden))
                # final_output = np.ravel(self.sigmoid(np.dot(hidden_output, weights_output) + bias_output))

                # #Calcolo della Binary Crossentropy 
                # loss = self.binary_crossentropy(tr_targets[i], final_output)
                # loss_tot.append(loss) # serve per il return
             
                # # Calcolo della derivata della Binary Crossentropy rispetto a y_pred
                # d_loss = self.binary_crossentropy_derivative(tr_targets[i], final_output)
                
                # # Backpropagation
                # d_output = d_loss * final_output * (1 - final_output)
                # # d_output = [d_output]
    
                # d_hidden = d_output * weights_output * hidden_output * (1 - hidden_output)
                
        
    
                # # Calcolo dei gradienti e aggiornamento dei pesi
                # weights_output = self.update_weights(np.ravel(weights_output), hidden_output*d_output, self.learningRate)
                # bias_output = self.update_weights(bias_output, d_output , self.learningRate)
                # # weights_hidden = self.update_weights(weights_hidden, tr_data[i] * d_hidden), self.learningRate)
                # bias_hidden = self.update_weights(bias_hidden, np.sum(d_hidden, axis=0, keepdims=True), self.learningRate)

        #     