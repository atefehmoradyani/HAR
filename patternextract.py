import numpy as np
import matplotlib.pyplot as plt
from numpy import unravel_index
# import tensorflow as tf
import os
import pywt
from numpy import unravel_index
from keras.applications import ResNet101
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input
from keras import Model
import cv2
import pandas as pd

base_model = ResNet101(weights='resnet101_weights_tf_dim_ordering_tf_kernels.h5')
# base_model.load_weights('resnet101_weights_tf_dim_ordering_tf_kernels.h5')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
model = base_model
# print(model.summary())

file = '15020.csv'
features = []
output_resnet101_dim = 1000
keyframe_num = 8
features.append(np.zeros([1, output_resnet101_dim*keyframe_num]))
print(file)
arr = []
arr = pd.read_csv('maxvalue_hmdb/' + file, header=None)
arr = np.array(arr)
value = arr[:, 3]
e = []
##
for item in value:
    e.append(float(item))
data = []
i = 0
while i < len(e):
    data.append(e[i])
    i += 15
plt.plot(data)  # doctest: +SKIP
plt.show()  # doctest: +SKIP

# print(len(data))
# print("len data = ", len(data))
coef, freq = pywt.cwt(data, 1, 'mexh')
# print(coef)
plt.plot(coef[0])  # doctest: +SKIP
plt.show()  # doctest: +SKIP

mean = sum(abs(coef[0]))/len(coef[0])
# mean = min(abs(coef[0]))
# print(mean)
# print(len(coef[0]))
pattern = []
start = -1
for i in range(len(coef[0])):
    # if start == -1 and abs(coef[0, i]) > mean:
    if start == -1:
        start = i
    elif start != -1 and abs(coef[0, i]) < mean:
        pattern.append([start, (i-1)])
        start = -1
    elif i == len(coef[0])-1:
        if start == -1:
            start = i-1
        pattern.append([start, i])
pattern = np.array(pattern)
# print(pattern)
# exit()
mergepattern = []
i = 0
start = -1
while i < len(pattern):
    if start == -1 and pattern[i, 1] - pattern[i, 0] < 8:
        start = pattern[i, 0]
        # print('ok')
    elif start != -1 and pattern[i, 1] - start >= 8:
        mergepattern.append([start, pattern[i, 1]])
        start = -1
    elif start == -1 and pattern[i, 1] - pattern[i, 0] >= 8:
        mergepattern.append(pattern[i])
    if i == len(pattern)-1 and pattern[i, 1] - pattern[i, 0] < 8:
        # print(pattern)
        if mergepattern != []:
            mergepattern[len(mergepattern)-1][1] = pattern[i, 1]
        else:
            mergepattern.append([start, pattern[i, 1]])
    i += 1
    # print('start = {}, i = {}'.format(start, i))
mergepattern = np.array(mergepattern)
# print(mergepattern)

if len(mergepattern) > 5:
    while len(mergepattern) > 5:
        lenmp = []
        for item in mergepattern:
            lenmp.append(item[1]-item[0]+1)
        lenmp = np.array(lenmp)
        # print("lenmp = ", lenmp)
        row = unravel_index(lenmp.argmin(), lenmp.shape)
        # print("row = ", row)
        row = row[0]
        # print("row = ", row)
        if 0 < row < len(mergepattern) - 1:
            if lenmp[row + 1] > lenmp[row - 1]:
                mergepattern[row] = [mergepattern[row, 0], mergepattern[row + 1, 1]]
                mergepattern = np.delete(mergepattern, row + 1, 0)
            elif lenmp[row+1] <= lenmp[row - 1]:
                mergepattern[row] = [mergepattern[row - 1, 0], mergepattern[row, 1]]
                mergepattern = np.delete(mergepattern, row - 1, 0)
        else:
            if row == 0:
                mergepattern[row] = [mergepattern[row, 0], mergepattern[row + 1, 1]]
                mergepattern = np.delete(mergepattern, row + 1, 0)
            elif row == len(lenmp)-1:
                mergepattern[row] = [mergepattern[row - 1, 0], mergepattern[row, 1]]
                mergepattern = np.delete(mergepattern, row - 1, 0)
print("mergepattern = ", mergepattern)
filename = file.split('.')
# exit()
video = filename[0] + '.avi'
cap = cv2.VideoCapture('videos_hmdb/' + video)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(frame)
frames = np.array(frames)
print("frame shape = ", frames.shape)
# print("keyframes shape = ", len(keyframes))

# for item in mergepattern:
#     keyframes = []
#     i = 1
#     episod = data[item[0]:item[1]]
#     # print(episod)
#     episod = np.array(episod)
#     # print('episod', episod.shape)
#     idx = 1
    # while idx <= 8:
    #     # print(episod)
    #     keyframe = unravel_index(episod.argmax(), episod.shape)
    #     keyframe = keyframe[0]
    #     episod[keyframe] = 0
    #     # print("keyframe = ", keyframe + item[0])
    #     keyframes.append([keyframe + item[0]])
    #     idx += 1
#     i += 1
#     feature = []
#     keyframes = np.sort(keyframes, axis=0)
#     for keyframe in keyframes:
#         # keyframe = keyframe[1]
#         input_x = frames[keyframe + 5, :, :, :]
#         img = np.resize(input_x, (224, 224, 3))
#         x = np.expand_dims(img, axis=0)
#         x = preprocess_input(x)
#         export = model.predict(x)
#         feature.append(export)
#     feature = np.array(feature)
#     feature = feature.reshape(1, output_resnet101_dim*keyframe_num)
#     features.append(feature)
# features = np.array(features)