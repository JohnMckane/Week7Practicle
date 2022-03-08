
import matplotlib.pyplot as plt
M = 2
C = -1
x, y = datasets.make_moons(500, noise=0.1)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)
epochs = 1000
alpha = 0.01

def Sigmoid(x):
    return 1/(1+np.exp(-x))
def dsdx(x):
        return Sigmoid(x)*(1-Sigmoid(x))
def loss(x,act):
    return 0.5*((x-act)**2)
def dlossdx(x,act):
    return x-act
class Network:
    def FeedForward(self,x):
        pass
    def FeedBackward(self,x,acc,alpha):
        pass
class Neuron1(Network):
    def CalculateInput(self,x):
        return (x[0]*self.w1) + (x[1]*self.w2)+self.b
    def __init__(self):
        self.w1 = np.random.rand()
        self.w2 = np.random.rand()
        self.b = np.random.rand()
    def FeedForward(self,x):
        return Sigmoid(self.CalculateInput(x))
class L2Neuron(Neuron1):
    def FeedBackward(self,x,acc,alpha):
        out = self.FeedForward(x)
        derrdout = out-acc
        doutdin = dsdx(self.CalculateInput(x))
        dindw1 = x[0]
        dindw2 = x[1]
        self.w1 = self.w1 - derrdout * doutdin * dindw1 * alpha
        self.w2 = self.w2 - derrdout * doutdin * dindw2 * alpha
        self.b = self.b - derrdout * doutdin * alpha
class L1(Network):
    def __init__(self,child):
        self.child=child
        self.w1 = np.random.rand()
        self.w2 = np.random.rand()
        self.w3 = np.random.rand()
        self.w4 = np.random.rand()
        self.b1 = np.random.rand()
        self.b2 = np.random.rand()

    def Out(self,x):
        i1,i2 = self.CalculateInput(x)
        return(Sigmoid(i1),Sigmoid(i2))
    def FeedForward(self,x):
        return self.child.FeedForward(self.Out(x))
    def CalculateInput(self,x):
        return (x[0]*self.w1 + x[1]*self.w2 + self.b1, x[0]*self.w3 + x[1]*self.w4 + self.b2)
    def FeedBackward(self,x,acc,alpha):
        ol1 = self.Out(x)
        self.child.FeedBackward(ol1,acc,alpha)
        dedo3 = self.child.FeedForward(ol1) - acc
        do3di3 = dsdx(self.child.CalculateInput(ol1))
        di3do1 = self.child.w1
        di3do2 = self.child.w2
        l1i = self.CalculateInput(x)
        do1di1 = dsdx(l1i[0])
        do2di2 = dsdx(l1i[1])
        di1dw1 = x[0]
        di1dw2 = x[1]
        di2dw3 = x[0]
        di2dw4 = x[1]
        self.w1 = self.w1 - alpha * dedo3 * do3di3 * di3do1 * do1di1 * di1dw1
        self.w2 = self.w2 - alpha * dedo3 * do3di3 * di3do1 * do1di1 * di1dw2
        self.b1 = self.b1 - alpha * dedo3 * do3di3 * di3do1 * do1di1
        self.w3 = self.w3 - alpha * dedo3 * do3di3 * di3do2 * do2di2 * di2dw3
        self.w4 = self.w4 - alpha * dedo3 * do3di3 * di3do2 * do2di2 * di2dw4
        self.b2 = self.b2 - alpha * dedo3 * do3di3 * di3do2 * do2di2


def PTEL(ntp,y,step):
    print("{} Epochs, {} Samples".format(len(ntp),len(ntp[0])))
    for i in range(0,len(ntp)):
        sli = 0
        for j in range(0,len(ntp[i])):
            sli = sli + loss(ntp[i][j],y[j])
        if(i % step == 0):
            print("Epoch {} Loss: {}".format(i,sli))
def CM(ntp,y,step,t,e):
    print("{} Epochs, {} Samples".format(len(ntp),len(ntp[0])))
    for i in range(0,e):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for j in range(0,len(ntp[i])):
            p = ntp[i][j] >= t

            if(y[j] == p):
                if(y[j] == 1):
                    TP = TP + 1
                else:
                    TN = TN + 1
            else:
                if(y[j] == 1):
                    FP = FP + 1
                else:
                    FN = FN + 1
        if(i % step == 0):
            print("TP {}, TN {}, FP {}, FN{}".format(TP,TN,FP,FN))
def Train(network,epochs,trdx,trdy,tedx,a):
    ntrp = [[0 for i in range(0,len(trdy))] for i in range(0,epochs)]
    ntep = [[0 for i in range(0,len(tedx))] for i in range(0,epochs)]
    for j in range(0,epochs):
        for i in range(0, len(trdy)):
            ntrp[j][i] = network.FeedForward(trdx[i])
            network.FeedBackward(trdx[i],trdy[i],a)
        for i in range(0,len(tedx)):
            ntep[j][i] = network.FeedForward(tedx[i])

    return (ntrp, ntep)
def T1NodeNetwork():
    #Try out a L2 Neuron on a data set
    n1 = L2Neuron()
    ntp, ntep = Train(n1,10000,x_train,y_train,x_test,0.001)
    #print Total Epoch Loss
    PTEL(ntp,y_train,1000)
    print(len(ntep))
    CM(ntep,y_test,1000,0.5,1000)
#Try 3 node network on a data set
l2 = L2Neuron()
l1 = L1(l2)
ntp, ntep = Train(l1,1000,x_train,y_train,x_test,0.001)
PTEL(ntp,y_train,100)
print(len(ntep))
CM(ntep,y_test,100,0.5,1000)










