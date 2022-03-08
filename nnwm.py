import numpy as np
import sklearn.model_selection
from sklearn import preprocessing
from sklearn import datasets
import mlutils
x, y = datasets.make_moons(1000, noise=0.01)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
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

        return np.multiply(self.didx() , self.dodi(self.CalculateInput(x)))
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
                print(dedo.shape)
                print(self.b.shape)
                raise (NameError("b shape change"))
            if(dedo.shape == () and dodi.shape == (1,1)):
                wd = dedo * dodi[0,0] * didw
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

fl = NetworkLayer(4,1,Sigmoid,dsdx)
n = NetworkLayer(4,4,relu,drdx,fl)
n2 = NetworkLayer(2,4,relu,drdx,NetworkLayer(4,4,relu,drdx,NetworkLayer(4,4,relu,drdx,NetworkLayer(4,4,relu,drdx,n))))
mlutils.MiniBatchTrain(n2,10,1000,x_train,y_train,x_test,y_test,0.1,0)




