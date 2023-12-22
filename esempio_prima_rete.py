# QUESTO È SOLO UN ESEMPIO, NON FA PARTE DEL PROGETTO!!


import math as mt
import random as rd 


# non ho ben capito seed
rd.seed(1)


def RN(m1,m2):
    t = m1*w1+m2*w2+b
    return sigmoide(t)

def sigmoide(t):
    return 1/(1+mt.exp(-t))


dataset=[[9,7.0,0],[2,5.0,1],[3.2,4.94,1],[9.1,7.46,0],[1.6,4.83,1],
        [8.4,7.46,0],[8,7.28,0],[3.1,4.58,1],[6.3,9.14,0],[3.4,5.36,1]]

# derivata della sigmoide 
def sigmoide_p(t):
    return sigmoide(t)*(1-sigmoide(t))

def train(): 
    # pesi pseudocasuali 
    w1=rd.random()
    w2=rd.random()
    b=rd.random()

    iterazioni = 10000
    learning_rate = 0.1 

    for i in range(iterazioni):
        
        # prendo indice casuale di un elemento nel dataset
        ri = rd.randint(0,len(dataset)-1)
        # prendo elemento corrispondente
        point = dataset[ri]

        z = point[0]* w1 +point[1] * w2 + b 
        pred = sigmoide(z)

        target = point[2]

        # costo 
        cost = (pred-target)**2 

        # derivate 
        dcost_dpred = 2 * (pred - target)
        dpred_dz = sigmoide_p(z)
        dz_dw1 = point[0]
        dz_dw2 = point[1]
        dz_db = 1

        dcost_dz = dcost_dpred * dpred_dz

        # derivate con regola della catena
        dcost_dw1 = dcost_dz * dz_dw1 
        dcost_dw2 = dcost_dz * dz_dw2 
        dcost_db = dcost_dz * dz_db

        # aggiorno pesi e bias usando il learning rate 
        w1 = w1 - learning_rate * dcost_dw1 
        w2 = w2 - learning_rate * dcost_dw2 
        b = b - learning_rate * dcost_db

    return w1,w2,b 

w1, w2, b = train()


# controllo 
pred = []
for gatto in dataset:
    z = w1 * gatto[0] + w2 * gatto[1] + b
    prediction = sigmoide(z)
    if prediction <= 0.5:
        pred.append('giungla')
    else: 
        pred.append('sabbie')

print(pred)