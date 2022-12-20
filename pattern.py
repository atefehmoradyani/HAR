import numpy as np
import matplotlib.pyplot as plt
from numpy import unravel_index
# import tensorflow as tf
import os
import pywt
from numpy import unravel_index
from keras.applications import ResNet101
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input
from keras import Model
import cv2
import pandas as pd
import sys

np.set_printoptions(threshold=sys.maxsize)
base_model = VGG16(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
model = Model(inputs=base_model.input, outputs=base_model.output)

# base_model = ResNet101(weights='resnet101_weights_tf_dim_ordering_tf_kernels.h5')
# base_model.load_weights('resnet101_weights_tf_dim_ordering_tf_kernels.h5')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
# model = base_model
# print(model.summary())
print(model.summary())
lst = os.listdir('maxvalue_hmdb')
lst2 = os.listdir('input_DAE_hmdb')
for file in lst:
    if file not in lst2:
        # print(file)
    # if os.path.exists("input_DAE/" + file):
        #     continue
        features = []
        # output_resnet101_dim = 1000
        keyframe_num = 8
        features.append(np.zeros([1, 1000*keyframe_num]))
        print(file)
        arr = pd.read_csv('maxvalue_hmdb/' + file, header=None)
        arr = np.array(arr)
        value = arr[:, 3]
        e = []
        for item in value:
            e.append(float(item))
        data = []
        i = 0
        while i < len(e):
            data.append(e[i])
            i += 15
        coef, freq = pywt.cwt(data, 1, 'mexh')
        mean = sum(abs(coef[0]))/len(coef[0])
        pattern = []
        start = -1
        for i in range(len(coef[0])):
            if start == -1 and abs(coef[0, i]):
                start = i
            elif start != -1 and abs(coef[0, i]) < mean:
                pattern.append([start, (i-1)])
                start = -1
            elif i == len(coef[0])-1:
                if start == -1:
                    start = i-1
                pattern.append([start, i])
        pattern = np.array(pattern)
        # print("1")

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
        # print("2")

        if len(mergepattern) > 5:
            while len(mergepattern) > 5:
                lenmp = []
                for item in mergepattern:
                    lenmp.append(item[1]-item[0]+1)
                lenmp = np.array(lenmp)
                # print("lenmp = ", lenmp)
                row = unravel_index(lenmp.argmin(), lenmp.shape)
                # print("row = ", row[0])
                # exit()
                # row = row[0]
                # print("row = ", row)
                row = row[0]
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
        # print("mergepattern = ", mergepattern)

        filename = file.split('.')
        video = filename[0] + '.avi'
        cap = cv2.VideoCapture('videos_hmdb/' + video)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            # print("3")
        frames = np.array(frames)
        # print("4")
        # print("frame shape = ", frames.shape)
        # print("keyframes shape = ", len(keyframes))
        # print(mergepattern)
        for item in mergepattern:
            keyframes = []
            i = 1
            episod = data[item[0]:item[1]]
            episod = np.array(episod)
            idx = 1
            epi = np.array(episod)
            if len(epi) < 8:
                continue
            while idx <= 8:
                x = np.where(epi == np.amax(epi))
                x = x[0][0]
                if np.amax(epi) != 0:
                    # print(x)
                    keyframe = x
                    epi[keyframe-1:keyframe+2] = 0
                    if x == 0:
                        epi[keyframe] = 0
                        epi[keyframe+1] = 0
                    if x == len(epi)-1:
                        epi[keyframe] = 0
                        epi[keyframe - 1] = 0
                    keyframes.append([keyframe + item[0]])
                    # print(keyframes)
                    idx += 1
                else:
                    # print(keyframes)
                    for epiIndex in keyframes:
                        episod[epiIndex-item[0]] = 0
                    epi = np.array(episod)
                # print("keyframe = ", keyframe + item[0])
                # print(idx)
            i += 1
            feature = []
            # print("keyframes = {}".format(keyframes))
            # print("sorted keyframes = {}".format(np.sort(keyframes, axis=0)))
            keyframes = np.sort(keyframes, axis=0)
            for keyframe in keyframes:
                # print(keyframe + 5)
                input_x = frames[keyframe + 5, :, :, :]
                img = np.zeros((224, 224, 3))
                for i in range(3):
                    img[:, :, i] = cv2.resize(input_x[0, :, :, i], (224, 224))
                x = np.expand_dims(img, axis=0)
                # print(x[0, :, :, :])
                # exit()
                # x = x/255
                # print("---------------")
                # print(x)
                export = model.predict(x)
                # print(export)
                # exit()
                feature.append(export)
            feature = np.array(feature)
            feature = feature.reshape(1, 1000*keyframe_num)
            features.append(feature)
            # print(feature)
            # print("================")
        features = np.array(features)
        print("features len = ", features.shape)
        np.savetxt("input_DAE_hmdb/" + file, features[:, 0, :], delimiter=',')
