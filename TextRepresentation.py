from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
train = ['Well done!',
         "Very Very Good",
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
labels_train = np.array([1,1,1,1,1,1,0,0,0,0,0])

test = ['Amazing job!',
        'Fantastic work',
        'Good effort',
        'Could not have done better',
        'not great',
        'poor job',
        'very weak',]

labels_test = np.array([1,1,1,1,0,0,0])
vectorizer = TfidfVectorizer()
data_train = vectorizer.fit_transform(train).toarray()
data_test = vectorizer.fit_transform(test).toarray()
print(data_train)
print(data_test)
model = Sequential()
model.add(Dense(10,input_dim=len(data_train[0]),activation="sigmoid"))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(data_train,labels_train,epochs=50,batch_size=2)
model.evaluate(data_test,labels_test)
