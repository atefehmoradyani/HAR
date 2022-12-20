import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow.python.keras
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, Reshape, Input, InputLayer, AveragePooling3D
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras import optimizers
from numpy.linalg import norm
import os

## use save model
model = load_model('modelucf2.h5')
model = load_model('model_hmdb_2.h5')
print(model.summary())
# print(model.layers[-3])
encoded = model.layers[-2]
encoder = Model(model.input, encoded.output)
print(encoder.summary())

# lst = os.listdir('features')
lst = os.listdir('features_hmdb_1')
for file in lst:
    data = np.array(np.zeros((1, 1000)))
    print(file)
    input_data = np.zeros((1, 5000))
    # fi = pd.read_csv('features/' + file, header=0)
    fi = pd.read_csv('features_hmdb_1/' + file, header=0)
    fi = np.array(fi)
    fi = np.reshape(fi, (1, len(fi)*1000))
    input_data[0, 0: fi.shape[1]] = fi
    # input_data = np.reshape(input_data, (1, 5000))
    # data = preprocessing.normalize(data, norm='l2')
    data = np.append(data, encoder.predict(input_data), axis=0)
    # np.savetxt('features2/' + file, data, delimiter=',')
    np.savetxt('features_hmdb_2/' + file, data, delimiter=',')


