import numpy as np
import cv2
import math
from sklearn import preprocessing
from numpy import unravel_index
import time
import os
def Dollar(video):
    video_read = "videos_hmdb/" + video
    cap = cv2.VideoCapture(video_read)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('img2', gray)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
        frames.append(gray)
    # varaibels -------------------
    bound_spatial = 9
    bound_temporal = 11
    sigma = 2.4
    tau = 1.7
    omega = 1 / tau
    number_of_R_value = 15
    NumberFrame = len(frames)
    bt = int((bound_temporal - 1) / 2)
    bs = int((bound_spatial - 1) / 2)
    width = len(frames[0][0])
    height = len(frames[0])
    # print('width: {}, height: {}'.format(width, height))
    # --------------------
    # Gaussian
    frames = np.array(frames)
    print(frames.shape)
    for f in range(NumberFrame):
        frames[f, :, :] = cv2.GaussianBlur(frames[f, :, :], (bound_spatial, bound_spatial), sigma)
        # cv2.imwrite('blure/img' + str(f) + '.jpg', frames[f, :, :])
    #--------------
    # gabor filter
    Hev = np.zeros((1, bound_temporal))
    Hod = np.zeros((1, bound_temporal))
    ti = range(-5, 6, 1)
    ti = np.array(ti)
    for h in range(bound_temporal):
        Hev[0, h] = -1 * math.cos(2 * math.pi * ti[h] * omega) * math.exp(-1 * (ti[h] ** 2) / (tau ** 2))
        Hod[0, h] = -1 * math.sin(2 * math.pi * ti[h] * omega) * math.exp(-1 * (ti[h] ** 2) / (tau ** 2))
    # Hev = preprocessing.normalize(Hev, norm='l1')
    # Hod = preprocessing.normalize(Hod, norm='l1')
    # frames = np.array(frames, dtype='f')
    rarry = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]))
    maxarray = []

    # print(frames[10-bt:10+bt, 10-bs:10+bs, 10-bs:10+bs])

    for f in range(NumberFrame):
        if bt <= f < NumberFrame - bt:
            for h in range(height):
                # if bs <= h < height - bs:
                for w in range(width):
                    # if bs <= w < width - bs:
                    rarry[f, h, w] = (sum(Hev[0] * frames[(f-bt):(f+bt+1), h, w]) ** 2) + (sum(Hod[0] * frames[(f-bt):(f+bt+1), h, w]) ** 2)
                    # maxarray.append([f, h, w, rarry[f, h, w]])
            # find max R
            img = rarry[f, :, :]
            idx = 0
            while idx < number_of_R_value:
                h, w = np.where(img == np.amax(img))
                value = img[h[0], w[0]]
                # print("f = {}, h = {}, w ={}, value = {}".format(f, h, w, value))
                maxarray.append([f, h[0], w[0], value])
                img[(h[0] - 1):(h[0] + 2), (w[0] - 1):(w[0] + 2)] = 0
                idx += 1

    maxarray = np.array(maxarray)
    video = video.split('.')[0]
    np.savetxt("maxvalue_hmdb/" + video + ".csv", maxarray, delimiter=",")
    return maxarray


path = "videos_hmdb/"
lst = os.listdir(path)
lst2 = os.listdir('maxvalue_hmdb/')
# for file in lst:
    # if file == '1067.avi':
    # print(file)
    # if file.split('.')[0]+'.csv' not in lst2:
file = '1075.avi'
digit = [int(d) for d in file.split('.')[0]]
No = digit[len(digit)-3]*100 + digit[len(digit)-2]*10 + digit[len(digit)-1]
# if No <= 20:
start_time = time.time()
print(file)
Dollar(file)
print("--- %s se conds ---" % (time.time() - start_time))
