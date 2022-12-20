import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow.python.keras
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, Reshape, Input, InputLayer, AveragePooling3D
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras import optimizers
from numpy.linalg import norm
import os

np.random.seed(45)

## concatenate  all data
# lst = os.listdir('input_DAE')
# data = []
# for file in lst:
#     print(file)
#     data.append(pd.read_csv('input_DAE/' + file, header=0))
# c = pd.concat(data)
# print("len c = ", len(c))
# np.savetxt('in_DAE.csv', c, delimiter=',')


## design model
# sgd = optimizers.SGD(lr=0.005, momentum=0.1, decay=0.1)
input_img = Input(shape=(8000,))
# encoded = Dense(4000, activation='sigmoid')(input_img)
# encoded = Dense(2000, activation='sigmoid')(encoded)
encoded = Dense(1000, activation='relu')(input_img)
# decoded = Dense(2000, activation='sigmoid')(encoded)
# decoded = Dense(8000, activation='sigmoid')(encoded)
decoded = Dense(8000, activation='softmax')(encoded)
autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['cosine_proximity'])
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.summary()

lst = os.listdir('input_DAE_hmdb')
id = 1
data = []
for file in lst:
    print(file)
    input_data = pd.read_csv('input_DAE_hmdb/' + file, header=0)
    input_data = np.array(input_data)
    input_data = np.round(input_data, decimals=5)
    # input_data = np.round(input_data)
    for item in input_data:
        data.append(item)

data = np.array(data)
# x_train = pd.read_csv('in_DAE.csv', header=0)
# x_train = np.array(x_train)
# x_train = preprocessing.normalize(x_train, norm='l2')
autoencoder.fit(data, data, epochs=10000, batch_size=128)
autoencoder.save('modelhmdb1.h5')


## use save model
# model = load_model('model')
# print(model.summary())
# print(model.layers[-3])
# encoded = model.layers[-3]
# encoder = Model(model.input, model.layers[-3].output)
# print(encoder.summary())
# x = x_train[100, :].reshape(1, 8000)
# print(encoder.predict(x))