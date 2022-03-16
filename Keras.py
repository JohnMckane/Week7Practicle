from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
data = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
x = data[:,0:8]
y = data[:,8]
model = Sequential()
model.add(Dense(12,input_dim=8,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
print("Model Compiled")
model.fit(x,y,epochs=1000,batch_size=30)
print(model.evaluate(x,y))
