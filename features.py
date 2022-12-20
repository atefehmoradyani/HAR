import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow.python.keras
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, Reshape, Input, InputLayer, AveragePooling3D
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras import optimizers
from numpy.linalg import norm
import os

# x = np.array([[1, 2, 3], [4, 5, 6]])
# print(x.shape)
# print(np.reshape(x, 6, order='F'))
# exit()
## use save model
# model = load_model('modelucf1.h5')
model = load_model('modelhmdb1.h5')
print(model.summary())
# print(model.layers[-3])
encoded = model.layers[-2]
encoder = Model(model.input, encoded.output)
print(encoder.summary())

# lst = os.listdir('input_DAE')
lst = os.listdir('input_DAE_hmdb')
for file in lst:
    data = np.zeros((1, 1000))
    # data = []
    # data = np.array(data)
    print(file)
    # input_data = pd.read_csv('input_DAE/' + file, header=0)
    input_data = pd.read_csv('input_DAE_hmdb/' + file, header=0)

    # print(len(input_data))
    input_data = np.array(input_data)
    for vector in input_data:
        vector = vector.reshape(1, 8000)
        data = np.append(data, encoder.predict(vector), axis=0)
        # print(data.shape)

        # vector = vector.reshape(8, 1000)
        # conv3d = []
        # conv3d = np.array(conv3d)
        # for i in range(1000):
        #     conv3d = np.append(conv3d, max(vector[:, i]))
        # data.append(conv3d)
    # np.savetxt('features/' + file, data, delimiter=',')
    np.savetxt('features_hmdb_1/' + file, data, delimiter=',')
    # np.savetxt('conv/' + file, data, delimiter=',')


