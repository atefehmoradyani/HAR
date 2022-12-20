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

    # ==========================================================
    # i = 1;
    # while (True):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #
    #     # Our operations on the frame come here
    #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #     # Display the resulting frame
    #     cv2.imshow('frame', frame)
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     cv2.imwrite('img/gray/' + str(i) + '.jpg', gray)
    #     i += 1
    #     if cv2.waitKey(0) & 0xFF == ord('q'):
    #         break
    #
    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # t = resize(frame, (200, 200, 3))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('img2', gray)

        # img = np.zeros((100, 100, 3))
        # for i in range(3):
        #     img[:, :, i] = cv2.resize(frame[:, :, i], (100, 100))
            # cv2.imshow('img', cv2.resize(frame[:, :, i], (224, 224)))
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break

        # print(img.shape)
        # img = img.astype(np.uint8)
        # cv2.imshow("img", img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        # cv2.imshow("img", img)
        # exit()
        frames.append(gray)
    # varaibels -------------------
    bound_spatial = 9
    bound_temporal = 11
    sigma = 2.4
    tau = 1.7
    omega = 1 / tau
    number_of_R_value = 1
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
        # cv2.imwrite('img/blure/' + str(f) + '.jpg', frames[f, :, :])
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
                for w in range(width):
                    rarry[f, h, w] = (sum(Hev[0] * frames[(f-bt):(f+bt+1), h, w]) ** 2) + (sum(Hod[0] * frames[(f-bt):(f+bt+1), h, w]) ** 2)

            # cv2.imwrite('img/gabor/' + str(f) + '.jpg', rarry[f, :, :])
            # find max R
            img = rarry[f, :, :]
            # print('f = ', f)
            # print('===============')

            idx = 0
            # img[0:bs+1, :] = 255
            # img[height-bs:, width-bs:] = 0
            # cv2.imwrite('action/img' + str(f) + '.jpg', img)
            # print(img[0:bs, 0:bs])
            while idx < number_of_R_value:
                h, w = np.where(img == np.amax(img))
                value = img[h[0], w[0]]
                # print("f = {}, h = {}, w ={}, value = {}".format(f, h, w, value))
                maxarray.append([f, h[0], w[0], value])
                img[(h[0] - 1):(h[0] + 2), (w[0] - 1):(w[0] + 2)] = 0
                idx += 1
        else:
             rarry[f, :, :] = 0

    # ==========================================================
    cap = cv2.VideoCapture(video_read)
    f = 0
    print('start:')
    while(True):
        ret, frame = cap.read()
        img2 = frame
        if ret == True:
            if bt <= f < len(frames)-bt:
                v = 0
                for item in maxarray:
                    if item[0] == f:
                        if v < 5:
                            if v == 1:
                                img2 = cv2.rectangle(frame, (item[2], item[1]), (item[2] + 3, item[1] + 3), (45, 255, 255), 3)
                                cv2.imshow('img2', img2)
                            else:
                                img2 = cv2.rectangle(frame, (item[2], item[1]), (item[2] + 3, item[1] + 3), (45, 255, 255), 3)
                                cv2.imshow('img2', img2)
                            v += 1
                        else:
                            break

            cv2.imwrite('img/dollar3/' + str(f) + '.jpg', img2)
            f += 1
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    return maxarray


path = "videos_hmdb/"
lst = os.listdir(path)
lst2 = os.listdir('maxvalue_hmdb/')
for file in lst:
    file = '15020' \
           '.avi'
    start_time = time.time()
    print(file)
    Dollar(file)
    print("--- %s seconds ---" % (time.time() - start_time))
