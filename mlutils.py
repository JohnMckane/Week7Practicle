import numpy as np
import sklearn.model_selection
from sklearn import preprocessing
from sklearn import datasets
def inttod(n,a):
    o = [0 for i in range(0,n)]
    o[a] = 1
    return np.matrix(o)
def Sigmoid(x):
    return 1/(1+np.exp(-x))
def dsdx(x):
        return Sigmoid(x)*(1-Sigmoid(x))
def loss(x,act):
    return 0.5*((x-act)**2)
def dlossdx(x,act):
    return x-act
def PTEL(ntp,y,step):
    print("{} Epochs, {} Samples".format(len(ntp),len(ntp[0])))
    for i in range(0,len(ntp)):
        sli = 0
        for j in range(0,len(ntp[i])):
            sli = sli + loss(ntp[i][j],y[j])
        if(i % step == 0):
            print("Epoch {} Loss: {}".format(i,sli))
def CBLoss(lf,network,x,y):
    return sum([lf(network.FeedForward(np.matrix(x[i])),y[i]) for i in range(0,len(y))])
def CBCM(network, x, y,t):
    p = [network.FeedForward(np.matrix(x[i])) > t for i in range(0,len(y))]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    TF = 0
    for i in range(0,len(y)):
        if(y[i] == p[i]):
            if(y[i] == 1):
                TP = TP + 1
            else:
                TN = TN + 1
        else:
            TF = TF +1
            if(y[i] == 1):
                FN = FN + 1
            else:
                FP = FP + 1
    print("TP {}, TN {}, FP {}, FN {}, TF {}".format(TP/len(y),TN/len(y),FP/len(y),FN/len(y), TF/len(y)))

def CM(ntp,y,step,t,e):
    print("{} Epochs, {} Samples".format(len(ntp),len(ntp[0])))

def Train(network,epochs,trdx,trdy,tedx,a):
    ntrp = [[0 for i in range(0,len(trdy))] for i in range(0,epochs)]
    ntep = [[0 for i in range(0,len(tedx))] for i in range(0,epochs)]
    for j in range(0,epochs):
        for i in range(0, len(trdy)):
            x = np.matrix(trdx[i])
            y = np.matrix(trdy[i])
            ntrp[j][i] = network.FeedForward(x)
            network.FeedBackward(x,y,a)
        for i in range(0,len(tedx)):
            x = np.matrix(trdx[i])
            ntep[j][i] = network.FeedForward(x)

    return (ntrp, ntep)

def MiniBatchTrain(network,nb,epochs,trdx,trdy,tedx,tedy,lf,a,s):
    #Calculate MiniBatch Starts and Ends
    bs = len(trdy)//nb
    bse =[i for i in range(0,len(trdy),bs)]
    el = CBLoss(lf,network,tedx,tedy)
    print(el)
    if(bse[len(bse)-1] != len(trdy)):
        bse.append(len(trdy))
    for i in range(0,epochs):
        for j in range(0, nb):
            bs = bse[j]
            be = bse[j+1]
            network.FeedBatchBackward(trdx[bs:be],trdy[bs:be],a)
        #Calculate Epoch Loss
        nel = CBLoss(lf,network,tedx,tedy)
        if(el - nel < s):
            print("Lim Reached")
            print(el)
            print(nel)
            break
        elif(i % (epochs//10) == 0):
            print(i)
            print(nel)
            CBLoss(lf,network,tedx,tedy)
        el = nel


def T1NodeNetwork(n1,x_train,y_train,x_test,y_test,ep=1000,a=0.1):
    #Try out a L2 Neuron on a data set
    print("Training Network")
    ntp, ntep = Train(n1,ep,x_train,y_train,x_test,a)
    #print Total Epoch Loss
    CM(ntp,y_train,ep/10,0.5,ep)
    print(len(ntep))
    print(len(y_test))
    CM(ntep,y_test,ep/10,0.5,ep)
