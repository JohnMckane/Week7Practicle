import mlutils
import numpy as np
import sklearn.model_selection
from sklearn import preprocessing
from sklearn import datasets
import mlutils
import pickle
n = mlutils.LoadNetwork("SoftMax")
print(n)
x,y = datasets.make_blobs(n_samples=5000, n_features=3, centers=4)
y=[mlutils.inttod(4,y[i]) for i in range(0,len(y))]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
for i in range(0, len(y_test)):
print(n.FeedForward(np.matrix(x_test[5])))
print(y_test[0])
