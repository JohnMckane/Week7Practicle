import numpy as np
import sklearn.model_selection
import tensorflow as tf
from sklearn import preprocessing
from sklearn import datasets
import matplotlib.pyplot as plt
M = 2
C = -1
x, y = datasets.make_moons(500, noise=0.1)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)
epochs = 1000
alpha = 0.01

def Sigmoid(x):
    return 1/(1+np.exp(-x))
def dsdx(x)
        return Sigmoid(x)*(1-Sigmoid(x))
def loss(x,act):
    return 0.5*((x-act)**2)
def dlossdx(x,act):
    return x-act
def neuron(i,a,w,b):
    return a(np.matmul(i,w) + b)
def dodi(i,d,w,b):
    return d(np.matmul(i,w)+b)


#Init Weights
w_1 = np.random.rand(2,3)
w_2 = np.random.rand(3,2)
w_3 = np.random.rand(2,1)
b_1 = np.random.rand(3)
b_2 = np.random.rand(2)
b_3 = np.random.rand(1)
a = Sigmoid
#train neural net
for i in range(0,epochs):
    for j in range(0, len(y_train)):
        #Calculate l1 out
        l1_out = neuron(x_train[j],a,w_1,b_1)
        #Calculate l2 out
        l2_out = neuron(l1_out,a,w_2,b_2)
        #Calculate l3 out
        l3_out = neuron(l2_out,a,w_3,b_3)
        #Calculate Error
        e = loss(l3_out,y_train[j])
        #adjust w_3
        dedo3 = dlossdx(l3_out)
        do3di3 = dodi(l2_out)
        w_3=w_3+alpha*(dedo2*do2di2*l2_out)




