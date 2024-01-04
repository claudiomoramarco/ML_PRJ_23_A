import numpy as np

#########################################################################################################

class Unit:

    # da aggiungere funzione di attivazione
    def __init__(self, isInput, isOutput, isHidden, precedentIsInput):
        if (isInput + isOutput > 1) or (isInput + precedentIsInput > 1) or (isInput+isOutput+isHidden == 0):
            print("Unit:__init__: errore")
            return
        
        # self.index = index # indice dell'unità, da usare nelle SD
        self.isInput = isInput
        self.isOutput = isOutput
        self.isHidden = isHidden
        self.precedentIsInput = precedentIsInput

        
    
    def initializeWeightsForUnit(self, numberInputUnits, numberUnitsForHLayer):
            # ogni unità mantiene un array dei pesi dei collegamenti delle unità del precedente layer con l'unità corrente 
            # ordinati e inizializzati random   
            # L'input layer non avrà pesi , tenerne conto
            if self.isInput:
                print("Unit:initializeWeightsForUnit:ERROR")
                return
            
            if self.precedentIsInput:
                self.weightsForUnit = np.random.randn(numberInputUnits)
            
            else:
                self.weightsForUnit = np.random.randn(numberUnitsForHLayer)


    # restituisce il valore di net e output
    # input è il vettore degli input 
    def computeUnitOutput(self, inputs): 
        if self.isInput: 
            net = inputs # singolo valore 
            output = net 
        
        else:
            if len(inputs) != len(self.weightsForUnit):
                print("Unit:computOutput:ERROR")
                exit      
            net = np.dot(inputs,self.weightsForUnit)
            output = net
    
        return(output,net)
        
            
        
#########################################################################################################


class Layer:



    def __init__(self, isInput, isOutput, isHidden, nUnits, precedentIsInput, numberInputUnits, numberUnitsForHLayer):
        if (isInput + isOutput > 1) or (isInput + precedentIsInput > 1) or (isInput+isOutput+isHidden==0):
            print("Layer:__init__: errore")
            return
        
        self.isInput = isInput
        self.isOutput = isOutput
        self.isHidden = isHidden
        self.nUnits = nUnits
        self.precedentIsInput = precedentIsInput
        
        #servono per sapere il numero di unità precedenti
        self.numberInputUnits = numberInputUnits
        self.numberUnitsForHLayer = numberUnitsForHLayer


        # CREAZIONE UNITS
        # array delle unità nel layer, dimensione self.nUnits
        self.units = [] 
        for i in range(self.nUnits):
            unit_new = Unit(self.isInput, self.isOutput, self.isHidden, self.precedentIsInput)
            if not unit_new.isInput:
                unit_new.initializeWeightsForUnit(self.numberInputUnits, self.numberUnitsForHLayer)

            self.units.append(unit_new)

        


    # restituisce il suo output e il suo net
    # oldSuccessiveNets mi serve per dopo per la backpropagation => forse non qui (per adesso non lo metto)
    def computeLayerOutput(self, oldPrecedentOutputs):
        
        # CALCOLO OUTPUT
        
        currentOutputs = []
        currentNets = []

        # se è input il precedentOutputs sarà il vettore tr_data
        if self.isInput:
            for i in range(len(oldPrecedentOutputs)):
                ret = self.units[i].computeUnitOutput(oldPrecedentOutputs[i])
                out = ret[0]
                net = ret[1]
                currentOutputs.append(out)
                currentNets.append(net)
            oldPrecedentOutputs = currentOutputs

        else:
            for unit in self.units:
                ret = unit.computeUnitOutput(oldPrecedentOutputs)  
                out = ret[0]
                net = ret[1]  
                currentOutputs.append(out)
                currentNets.append(net)
            oldPrecedentOutputs = currentOutputs

        return (currentOutputs,currentNets)


#########################################################################################################
        

class NN:

    # sto supponendo che ogni hidden layer abbia stesso numero di neuroni
    def __init__(self, numberInputUnits, numberOutputUnits, numberUnitsForHLayer, numberHiddenLayers):
        self.numberInputUnits = numberInputUnits
        self.numberOutputUnits = numberOutputUnits
        self.numberUnitsForHLayer = numberUnitsForHLayer
        self.numberHiddenLayers = numberHiddenLayers
        


    def run(self, tr_data, tr_targets, numberEpochs): 

        if (len(tr_data[0]) != self.numberInputUnits) or (len(tr_targets[0]) != self.numberOutputUnits):
            print("NN:run: ERROR")
            return

        # per adesso una sola epoca 
        if (numberEpochs != 1):
            print("NN:run: ERROR")
            return


        # CREAZIONE DEI LAYER 
        self.layers = []
        #inputLayer 
        self.layers.append(Layer(1,0,0,self.numberInputUnits,0,self.numberInputUnits,self.numberUnitsForHLayer))
        # creazione hidden layers 
        for i in range(self.numberHiddenLayers):
            if i == 0: # il primo 
                self.layers.append(Layer(0,0,1,self.numberUnitsForHLayer, 1, self.numberInputUnits, self.numberUnitsForHLayer))
            else: 
                self.layers.append(Layer(0,0,1,self.numberUnitsForHLayer, 0, self.numberInputUnits, self.numberUnitsForHLayer))
        #creazione output layer
        self.layers.append(Layer(0,1,0,self.numberOutputUnits,self.numberHiddenLayers == 0, self.numberInputUnits, self.numberUnitsForHLayer))
       
        # def computeLayerOutput(self, oldPrecedentOutputs, currentOutputs, currentNets):
        # ESECUZIONE DEI LAYER IN ORDINE => per ogni tr_data[i]

        # oldSuccessiveNets = [] => serve dopo per l'apprendimento
        # nets e outputs poi vanno salvati perché servono per la backpropagation
        precedentOutputs = []
        precedentNets = []  

        for i in range(len(tr_data)):
            #inputLayer 
            ret = self.layers[0].computeLayerOutput(tr_data[i])
            currentOutputs = ret[0]
            currentNets = ret[1]
            precedentOutputs = currentOutputs # per chiamare il prossimo layer
            precedentNets = currentNets

            # hidden layers e output layer (+1)
            for j in range(self.numberHiddenLayers+1):
                ret = self.layers[j+1].computeLayerOutput(precedentOutputs)
                currentOutputs = ret[0]
                currentNets = ret[1]
                precedentOutputs = currentOutputs # per chiamare il prossimo layer
                precedentNets = currentNets
    
                    
            # stampo l'output di ogni esempio per prova 
            print(currentOutputs)
                
#########################################################################################################



