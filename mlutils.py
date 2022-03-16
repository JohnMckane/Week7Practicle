import numpy as np
import sklearn.model_selection
from sklearn import preprocessing
from sklearn import datasets
import pickle
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
def Sanitise(x):
    if(x.shape == (1,1)):
        return x[0,0]
    else:
        return np.matrix(x)
def Sigmoid(x):
    return np.reciprocal(1+np.exp(-x))
def dsdx(x):
        return np.multiply(Sigmoid(x),1-Sigmoid(x))
def relu(x):
    lto = np.greater(x,0)
    return np.multiply(x,lto)
def drdx(x):
    return np.greater(x,0)
def loss(x,act):
    return 0.5*((x-act)**2)
def dlossdx(x,act):
    return x-act
class Network:
    def FeedForward(self,x):
        pass
    def FeedBackward(self,x,acc,alpha):
        pass
class NetworkLayer(Network):
    def __init__(self,si,so,Activation,dadi,child=None):
        self.Activation = Activation
        self.dadi = dadi
        self.si = si
        self.so = so
        self.w = np.random.rand(si,so)
        self.b = np.random.rand(1,so)
        self.child = child
    def CalculateInput(self,x):
        if(x.shape[1] != self.si):
            print(x.shape)
            print(self.si)
            raise(NameError("X Shape Incorrect"))
        inp = np.matmul(x,self.w) + self.b
        if(inp.shape != (1,self.so)):
            raise(NameError("inp Shape Wrong"))
        return Sanitise(inp)
    def FeedForward(self,x):
        lout = self.Output(x)
        if(lout.shape != (1,self.so)):
            raise(NameError("Output Shape Wrong"))
        if(self.child == None):
            return lout
        else:
            return self.child.FeedForward(lout)
    def Output(self,x):
        return Sanitise(self.Activation(self.CalculateInput(x)))
    def didw(self, x):
        return np.tile(np.transpose(x),self.so)
    def dodi(self,i):
        return Sanitise(self.dadi(i))
    def dodx(self,x):
        dodi =  self.dodi(self.CalculateInput(x))
        didx = self.didx()
        if(didx.shape[1] == dodi.shape[0]):
            return np.matmul(didx,dodi)
        return np.multiply(self.didx(),dodi)
    def didx(self):
        return self.w
    def dedo(self,o,act):
        if(self.child == None):
            return Sanitise(dlossdx(o,act))
        else:
            cs = self.child.dedo(self.child.Output(o),act)
            if(cs.shape == ()):
                return Sanitise(self.child.dodx(o) * cs)
            else:
                return Sanitise(np.matmul(self.child.dodx(o),cs))
    def FeedBackward(self,x,acc,alpha,batch=False):
            if(self.child != None and not batch):
                self.child.FeedBackward(self.Output(x),acc,alpha)
            dedo = self.dedo(self.Output(x),acc)
            dodi = self.dodi(self.CalculateInput(x))
            didw = self.didw(x)
            if(np.transpose(np.matrix(dedo)).shape != self.b.shape):
                print("dedo shape:{}".format(dedo.shape))
                print("b shape:{}".format(self.b.shape))
                raise (NameError("b shape change"))
            if(dedo.shape == () and dodi.shape == (1,1)):
                wd = dedo * dodi[0,0] * didw
            elif(dedo.shape[0] == dodi.shape[0] and didw.shape[1] == dodi.shape[1]):
                wd = np.transpose(np.matmul(np.multiply(dedo,dodi),np.transpose(didw)))
            else:
                wd = np.transpose(np.matmul(np.matmul(dedo,dodi),np.transpose(didw)))
            if(wd.shape != self.w.shape):
                print(wd.shape)
                print(self.w.shape)
                raise(NameError("W size change"))
            else:
                if(batch):
                    return np.matrix([alpha * wd,alpha * np.transpose(dedo)])
                else:
                    self.w = self.w - alpha * wd
                    self.b = self.b - alpha * np.transpose(dedo)
    def FeedBatchBackward(self,x,y,alpha):
        if(self.child != None):
            nx = [self.Output(np.matrix(x[i])) for i in range(0,len(x))]
            self.child.FeedBatchBackward(nx,y,alpha)
        d = [self.FeedBackward(np.matrix(x[i]),y[i],alpha,True) for i in range(0,len(y))]
        d = sum(d) * 1/len(y)
        self.w = self.w-d[0,0]
        self.b = self.b-d[0,1]
#Softmax Layer, using Cross-Entropy Loss Function
def SL(a):
    return np.log(np.maximum(a,np.ones(a.shape)*0.000001))
def SoftMax(i):
    ex = np.exp(i)
    return ex/np.sum(ex)
def dsoftdi(i):
    #using mult rule with u(x) = e^x and v(x) 1/sum(e^x)
    ex = np.exp(i)
    sex = np.sum(np.exp(i))
    #dvdi = dexdi * dsumdi * din/dsum
    dvdi = ex * (-sex**-2)
    dudi = ex
    u = np.transpose(ex)
    v = 1/sex
    return np.matmul(u, dvdi) + v * dudi
class SoftMaxLayer(NetworkLayer):
    def __init__(self,si,so):
        super().__init__(si,so,SoftMax,dsoftdi)
    def dedo(self,o,act):
        return - np.transpose(np.multiply(act,np.reciprocal(o)))
def CrossEntropy(q,p):
    lq = np.log(q)
    return -(np.sum(np.multiply(p,lq)))

def MakeSoftMaxNN(si,so,bw,a,dadi,depth):
    n = SoftMaxLayer(bw,so)
    for i in range(0,depth-2):
        n = NetworkLayer(bw,bw,a,dadi,n)
    return NetworkLayer(si,bw,a,dadi,n)
def SaveNetwork(n,name):
    with open('{}.network'.format(name), 'wb') as nf:
        pickle.dump(n, nf)
def LoadNetwork(name,ext=".network",loc=""):
    with open('{}.network'.format(name), 'rb') as nf:
        return pickle.load(nf)

