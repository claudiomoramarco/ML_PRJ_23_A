import numpy as np
import activation_functions
import sys

class Unit:

    # da aggiungere funzione di attivazione
    def __init__(self, isInput, isOutput, isHidden, precedentIsInput, learningRate, id, activation_function):
        if (isInput + isOutput > 1) or (isInput + precedentIsInput > 1) or (isInput+isOutput+isHidden == 0):
            print("Unit:__init__: errore")
            return
        
        # self.index = index # indice dell'unità, da usare nelle SD
        self.isInput = isInput
        self.isOutput = isOutput
        self.isHidden = isHidden
        self.precedentIsInput = precedentIsInput
        self.learningRate = learningRate
        self.id = id # unità numerate a partire da 0 per ogni layer
        self.weightsForUnit = []
        self.activationFunction = activation_function
        self.lastOut = 1
        self.lastNet = 1
        self.bias = 0 # inizia 0 (ok?)



    def initializeWeightsForUnit(self, numberInputUnits, numberUnitsForHLayer):
            # ogni unità mantiene un array dei pesi dei collegamenti delle unità del PRECEDENTE layer con l'unità corrente 
            # ordinati e inizializzati random   
            # L'input layer non avrà pesi , tenerne conto
            if self.isInput:
                print("Unit:initializeWeightsForUnit:ERROR")
                return
            
            if self.precedentIsInput:
                self.weightsForUnit = np.random.normal(0, 0.01, numberInputUnits) 
            else:
                self.weightsForUnit = np.random.normal(0, 0.01, numberUnitsForHLayer) 



    # restituisce il valore di net e output
    # input è il vettore degli input 
    def computeUnitOutput(self, inputs): 

        # print(inputs)
        # # print(len(inputs))
        
        if self.isInput:  # se è nell'input layer
            net = inputs # singolo valore 
            output = net 
        
        else:
            
            if len(inputs) != len(self.weightsForUnit): # +1 per il bias
                print("Unit:computOutput:ERROR")
                exit()      
                
            net = self.bias + np.dot(inputs,self.weightsForUnit)
            output = net
    
        # salvo l'ultimo net e l'ultimo output calcolato 
        self.lastNet = net # ultimo net calcolato
        self.lastOut = self.activationFunction(net) # ultimo output calcolato
        
        return(self.lastOut,self.lastNet)
    


    # target è il target corrispondente al neurone, serve solo nel caso dell'output layer
    # successiveUnits sono i neuroni del layer successivo, non servono per l'output layer
    def updateWeights(self, precedentUnits, successiveUnits,  target): 
        
            if self.isOutput:
                delta = (target-self.lastOut)*(activation_functions.derivative(self.activationFunction))(self.lastNet)
                self.lastDelta = delta # salvo l'ultimo delta calcolato
                for i in range(len(precedentUnits)): # per ogni neurone precedente
                    # if i == 0:
                        # print(self.weightsForUnit[i])
                        # print(self.learningRate)
                        # print(self.lastDelta)
                        # print(precedentUnits[i].lastOut)
                    self.weightsForUnit[i]+= self.learningRate*self.lastDelta*precedentUnits[i].lastOut
                # aggiorno il bias
                self.bias = self.learningRate * self.lastDelta

            else:
                tmp_deltaxW = 0
                for i in range(len(successiveUnits)): # per ogni peso del livello successivo
                    tmp_deltaxW +=  successiveUnits[i].lastDelta*successiveUnits[i].weightsForUnit[self.id] 
                tmp_deltaxW = tmp_deltaxW * (activation_functions.derivative(self.activationFunction))(self.lastNet)
                self.lastDelta = tmp_deltaxW
                for i in range(len(self.weightsForUnit)):
                    self.weightsForUnit[i]+= self.learningRate*self.lastDelta*self.lastOut
                if not self.isInput:
                    self.bias = self.learningRate * self.lastDelta
