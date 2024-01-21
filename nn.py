import activation_functions
import loss
import numpy as np

class NN:

    # layer sizes passato dall'esterno
    def __init__(self, layer_sizes, learningRate, activationFunctionForHidden, activationFunctionForOutput, lossFunction, isClassification):
        self.learningRate = learningRate
        self.isClassification = isClassification
        self.activationFunctionForHidden = activationFunctionForHidden
        self.activationFunctionForOutput = activationFunctionForOutput
        self.lossFunction = lossFunction
        self.layer_sizes = layer_sizes # contiene la lista del numero di neuroni di ogni layer
        self.weights = []
        self.biases = []
        self.initialize_weights()


#########################################################################################################
        
    # Inizializza pesi e bias in modo casuale
    def initialize_weights(self):
        
        np.random.seed(42)
       
        for i in range(len(self.layer_sizes)-1):
            
            self.weights.append(np.random.normal(0,  np.sqrt(2/self.layer_sizes[i]) ,(self.layer_sizes[i], self.layer_sizes[i+1])))
            self.biases.append(np.zeros((1, self.layer_sizes[i+1])))

#########################################################################################################

    def forward_pass(self, inputs):

        # Lista per salvare gli output di ogni layer
        self.layer_outputs = [inputs]
        # Input per il primo layer
        layer_input = inputs

        # Itera attraverso tutti i layer
        for i in range(len(self.layer_sizes)-1):
            # Calcola l'output per il layer corrente
            if i == len(self.layer_sizes)-2:
                layer_output = self.activationFunctionForOutput(np.dot(layer_input,self.weights[i]) + self.biases[i])
            else:   
                layer_output = self.activationFunctionForHidden(np.dot(layer_input, self.weights[i]) + self.biases[i])

            # Aggiungi l'output alla lista
            self.layer_outputs.append(layer_output)

            # L'output del layer corrente diventa l'input per il prossimo layer
            layer_input = layer_output

        return self.layer_outputs[-1]


#########################################################################################################

    def backward_pass(self, target):

        # Calcola il gradiente della loss rispetto all'output del layer di output
        output_gradient = loss.derivative(self.lossFunction)( target , self.layer_outputs[-1])

        # Itera all'indietro attraverso i layer per aggiornare i pesi
        for i in reversed(range(len(self.layer_sizes))):

            # Calcola il gradiente rispetto all'input del layer corrente
            if i == len(self.layer_sizes)-1:
                derivative_values = activation_functions.derivative(self.activationFunctionForOutput)(self.layer_outputs[i])
            else: 
                derivative_values = activation_functions.derivative(self.activationFunctionForHidden)(self.layer_outputs[i])


            derivative = output_gradient * derivative_values
        
            # Aggiorna i pesi e i bias del layer corrente
            self.update_weights_and_biases(derivative, self.layer_outputs[i], self.learningRate, i-1)

            # Calcola il gradiente per il layer precedente
            if i > 0:
                output_gradient = np.dot(derivative, np.transpose(self.weights[i-1]))

#########################################################################################################
    
    def run_training(self, tr_data, tr_targets, numberEpochs):  
        

        loss_epochs = []
        final_outputs = []

        for epoch in range(numberEpochs):

            epoch_loss = 0.0
            epoch_outputs = []

            # Itera su tutti i dati di training
            for i in range(len(tr_data)):
                # print(i)
                # Esegui il passaggio in avanti
                inputs = tr_data[i]
                targets = tr_targets[i]
                outputs = self.forward_pass(inputs)

                # Calcola la loss
                loss_value = self.lossFunction(targets, outputs)

                epoch_loss += loss_value

                # Esegui il passaggio all'indietro per aggiornare i pesi
                self.backward_pass(targets)

                # Aggiungi gli output correnti all'array degli output per l'epoca
                epoch_outputs.append(outputs)

            # Calcola la media della loss per l'epoca
            epoch_loss /= len(tr_data)
            loss_epochs.append(epoch_loss)
            final_outputs.append(np.ravel(epoch_outputs))

        return loss_epochs, final_outputs


#########################################################################################################
    

    def update_weights_and_biases(self, input_gradient, layer_output, learning_rate, index):
        
        # Calcola l'aggiornamento dei pesi
        weights_update = np.dot(np.transpose(layer_output), input_gradient)

        # Calcola l'aggiornamento dei bias
        biases_update = np.sum(input_gradient, axis=0, keepdims=True)
         
        # Applica l'aggiornamento dei pesi e dei bias
        clip_value = 1
        self.weights[index] -= learning_rate * np.clip(weights_update, -clip_value, clip_value)
        self.biases[index] -= learning_rate * np.clip(biases_update, -clip_value, clip_value)
