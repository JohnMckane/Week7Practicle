import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras import initializers
from keras.regularizers import l1
import matplotlib.pyplot as plt
import numpy as np
num_words=50000

df = pd.read_csv('yelp_reviews.csv',encoding = "ISO-8859-1")

#select input and output variables
data = df.values[:,0]
labels = df.values[:,1]
labels = np.asarray(labels).astype("float32")
train_data,test_data,train_labels,test_labels = train_test_split(data,labels,train_size=0.9)
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_data)
sequences = tokenizer.texts_to_sequences(train_data)
ml=max(map(len,sequences))
x_train_seq = pad_sequences(sequences, maxlen=ml)
print(test_data[0])
print(train_data[0])
print(test_labels[0])
print(train_labels[0])
sequences = tokenizer.texts_to_sequences(test_data)
x_test_seq = pad_sequences(sequences, maxlen=ml)
m1 = Sequential()
m1.add(Embedding(50000,300,input_length=ml,trainable=True))
m1.add(Conv1D(32,5,activation="relu"))
m1.add(MaxPooling1D())
init_he_u = initializers.he_uniform(seed=None)
m1.add(Dense(10,"relu",kernel_initializer=init_he_u,kernel_regularizer =l1(0.001)))
m1.add(Dense(1,"sigmoid",kernel_regularizer =l1(0.001)))
m1.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])
history = m1.fit(x_train_seq, train_labels, validation_data=(x_test_seq, test_labels), epochs=50, batch_size=32, verbose=2)
print(type(history.history["acc"]))
x = [i for i in range(0,len(history.history["acc"]))]
plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, history.history["acc"], color ="red")
plt.show()
