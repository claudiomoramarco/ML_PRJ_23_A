import numpy as np
import unit

class Layer:

    def __init__(self, isInput, isOutput, isHidden, nUnits, precedentIsInput, numberInputUnits, numberUnitsForHLayer, learningRate, activation_function):
        if (isInput + isOutput > 1) or (isInput + precedentIsInput > 1) or (isInput+isOutput+isHidden==0):
            print("Layer:__init__: errore")
            return
        
        self.isInput = isInput
        self.isOutput = isOutput
        self.isHidden = isHidden
        self.nUnits = nUnits
        self.precedentIsInput = precedentIsInput
        self.learningRate = learningRate

        #servono per sapere il numero di unità precedenti
        self.numberInputUnits = numberInputUnits
        self.numberUnitsForHLayer = numberUnitsForHLayer
        self.activationFunction = activation_function

        # CREAZIONE UNITS
        # array delle unità nel layer, dimensione self.nUnits
        self.units = [] 
        for i in range(self.nUnits):
            unit_new = unit.Unit(self.isInput, self.isOutput, self.isHidden, self.precedentIsInput, self.learningRate, i, self.activationFunction)
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



    # backpropagation lancia la funzione di aggiornamento dei pesi per ogni neurone del layer
    # alla fine della sua esecuzione i pesi di tutti i neuroni del layer sono aggiornati
    # Deve essere laciata da NN nel corretto ordine sui layer
    # precedent e successive Units sono le liste dei neuroni dei livelli precedente e successivo, 
    # nel caso output layer successiveUnits sarà nulla
    # targets è il vettore dei target dell'esempio considerato, serve solo nel caso dell'outputLayer
    def backpropagation(self,precedentUnits, successiveUnits,  targets ):
        # i pesi sono relativi ai collegamenti che arrivano dal precedente layer, quindi per l'input layer niente pesi
        if self.isInput:
            print("Layer:backpropagation:ERROR")
            exit()
            
        if self.isOutput:
            for i in range(len(self.units)):
                self.units[i].updateWeights(precedentUnits, None, targets[i])
        else:
            for i in range(len(self.units)):
                self.units[i].updateWeights(precedentUnits, successiveUnits,None)
        