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

def laf(m,x,c):
    res = m * x +c
    print(res)
    res = res - np.multiply(res-1,np.greater(res,1))
    res = res - np.multiply(res,np.less(res,0))
    print(res)
    return res
def dlafdx(m,x,c):
    res = m * x +c
    if(res <= 0):
        return 0
    if(res >= 1):
        return 0
    else:
        return m
def loss(x,act):
    return x-act
def dlossdx(x,act):
    return x
def neuron(i,a,w,b):
    return a(np.matmul(i,w) + b)
#Init Weights
w_1 = np.random.rand(2,3)
w_2 = np.random.rand(3,2)
w_3 = np.random.rand(2,1)
b_1 = np.random.rand(3)
b_2 = np.random.rand(2)
b_3 = np.random.rand(1)
a = lambda x: laf(M,x,C)
print(neuron(neuron(neuron(x_train[0],a,w_1,b_1),a,w_2,b_2),a,w_3,b_3))



