import numpy as np
import sklearn.model_selection
from sklearn import preprocessing
from sklearn import datasets
import mlutils
import pickle

x,y = datasets.make_blobs(n_samples=5000, n_features=3, centers=4)
y=[mlutils.inttod(4,y[i]) for i in range(0,len(y))]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)


n = mlutils.MakeSoftMaxNN(3,4,50,mlutils.Sigmoid,mlutils.dsdx,10)
mlutils.MiniBatchTrain(n,20,10,x_train,y_train,x_test,y_test,mlutils.CrossEntropy,0.00001,0)
mlutils.SaveNetwork(n,"SoftMax")






