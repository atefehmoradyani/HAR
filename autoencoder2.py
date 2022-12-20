import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow.python.keras
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, Reshape, Input, InputLayer, AveragePooling3D
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras import optimizers
from numpy.linalg import norm
import os

data = []
# lst = os.listdir('features')
lst = os.listdir('features_hmdb_1')
# lst = os.listdir('conv')
for file in lst:
    print(file)
    input_data = np.zeros((1, 5000))
    # fi = pd.read_csv('features/' + file, header=0)
    fi = pd.read_csv('features_hmdb_1/' + file, header=0)
    # fi = pd.read_csv('conv/' + file, header=None)
    fi = np.array(fi)
    fi = np.reshape(fi, (1, len(fi)*1000))
    input_data[0, 0: fi.shape[1]] = fi
    input_data = np.reshape(input_data, (1, 5000))
    data.append(input_data)
data = np.array(data)

## design model
# sgd = optimizers.SGD(lr=0.005, momentum=0.1, decay=0.1)
input_img = Input(shape=(5000,))
encoded = Dense(3000, activation='relu')(input_img)
# encoded = Dense(2000, activation='sigmoid')(encoded)
encoded = Dense(1000, activation='relu')(encoded)
# decoded = Dense(2000, activation='sigmoid')(encoded)
decoded = Dense(3000, activation='relu')(encoded)
decoded = Dense(5000, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'cosine_proximity'])
autoencoder.summary()

data = data[:, 0, :]
# data = preprocessing.normalize(data, norm='l1')
autoencoder.fit(data, data, epochs=50000, batch_size=128)
# autoencoder.save('modelucf2.h5')
autoencoder.save('model_hmdb_2.h5')


## use save model
# model = load_model('model')
# print(model.summary())
# print(model.layers[-3])
# encoded = model.layers[-3]
# encoder = Model(model.input, model.layers[-3].output)
# print(encoder.summary())
# x = x_train[100, :].reshape(1, 8000)
# print(encoder.predict(x))