import layer
import activation_functions
import loss

class NN:

    # sto supponendo che ogni hidden layer abbia stesso numero di neuroni
    def __init__(self, numberInputUnits, numberOutputUnits, numberUnitsForHLayer, numberHiddenLayers, learningRate, activationFunctionForHidden):
        self.numberInputUnits = numberInputUnits
        self.numberOutputUnits = numberOutputUnits
        self.numberUnitsForHLayer = numberUnitsForHLayer
        self.numberHiddenLayers = numberHiddenLayers
        self.learningRate = learningRate
        self.activationFunctionForHidden = activationFunctionForHidden




    def run_training(self, tr_data, tr_targets, numberEpochs): 

        if (len(tr_data[0]) != self.numberInputUnits) or (len(tr_targets[0]) != self.numberOutputUnits):
            print("NN:run_training: ERROR")
            return

        # salvo i targets 
        self.tr_targets = tr_targets

        # CREAZIONE DEI LAYER 
        self.layers = []
        #inputLayer 
        self.layers.append(layer.Layer(1,0,0,self.numberInputUnits,0,self.numberInputUnits,self.numberUnitsForHLayer, self.learningRate, activation_functions.linear))
        # creazione hidden layers 
        for i in range(self.numberHiddenLayers):
            if i == 0: # il primo 
                self.layers.append(layer.Layer(0,0,1,self.numberUnitsForHLayer, 1, self.numberInputUnits, self.numberUnitsForHLayer, self.learningRate, self.activationFunctionForHidden))
            else: 
                self.layers.append(layer.Layer(0,0,1,self.numberUnitsForHLayer, 0, self.numberInputUnits, self.numberUnitsForHLayer, self.learningRate, self.activationFunctionForHidden))
        #creazione output layer
        self.layers.append(layer.Layer(0,1,0,self.numberOutputUnits,self.numberHiddenLayers == 0, self.numberInputUnits, self.numberUnitsForHLayer, self.learningRate, activation_functions.linear))
       

        mseList = [] # lista dei mse, uno per ogni epoca

        for k in range(numberEpochs):


            epochoutput = [] # intero output dopo un'epoca
            
            # ESECUZIONE DEI LAYER IN ORDINE => per ogni tr_data[i]
            for i in range(len(tr_data)):
                
                #inputLayer 
                ret = self.layers[0].computeLayerOutput(tr_data[i])
                currentOutputs = ret[0]
                precedentOutputs = currentOutputs # per chiamare il prossimo layer

            
                # hidden layers e output layer (+1)
                for j in range(self.numberHiddenLayers+1):
                    ret = self.layers[j+1].computeLayerOutput(precedentOutputs)
                    currentOutputs = ret[0]
                    precedentOutputs = currentOutputs # per chiamare il prossimo layer
                # stampo l'output di ogni esempio per prova 
                # print("OUTPUT", i)
                # print(currentOutputs)
                
                epochoutput.append(currentOutputs)

                # APPRENDIMENTO 
                # ciclo inverso sui layers 
                # output layer 
                self.layers[self.numberHiddenLayers+1].backpropagation(self.layers[self.numberHiddenLayers].units, None, self.tr_targets[i])
                #hidden layers escluso input
                for j in range(self.numberHiddenLayers, 0, -1): 
                    self.layers[j].backpropagation(self.layers[j-1].units,self.layers[j+1].units,None)

            

            # CALCOLO E STAMPA DELLA LOSS
            meanSquared = loss.mse(self.tr_targets,epochoutput)
            print(meanSquared)
            mseList.append(meanSquared)
        
        return mseList


    # qua tutti i parametri della rete esistono già 
    def run_test(self, test_data, test_targets):
        
        print(len(test_data[0]))
        print( self.numberInputUnits)
        print(len(test_targets[0]))
        print(self.numberOutputUnits)        

        if (len(test_data[0]) != self.numberInputUnits) or (len(test_targets[0]) != self.numberOutputUnits):
            print("NN:run_test: ERROR")
            exit()
        
        # ESECUZIONE 
        outputs = []
        for i in range(len(test_data)):
            #inputLayer 
            ret = self.layers[0].computeLayerOutput(test_data[i])
            currentOutputs = ret[0]
            precedentOutputs = currentOutputs # per chiamare il prossimo layer

            # hidden layers e output layer (+1)
            for j in range(self.numberHiddenLayers+1):
                ret = self.layers[j+1].computeLayerOutput(precedentOutputs)
                currentOutputs = ret[0]
                precedentOutputs = currentOutputs # per chiamare il prossimo layer
            
            outputs.append(currentOutputs)
        
        return outputs