import numpy as np
import cv2
video_read = "11.mp4"
cap = cv2.VideoCapture(video_read)
frames = []
i = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(frame[:, :, ])
    # cv2.imshow('img', frame)
    cv2.imwrite('imgg/' + str(i) + '.jpg', frame)
    i = i + 1
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
#     # frames.append(gray)
# img = cv2.imread('filmf/30.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# im = np.array(gray)
# img = img[50:150, 250:300, :]
# im = cv2.resize(img, (500, 300))
# sampel = np.zeros(im.shape)
# print(sampel.shape)
# for i in range(sampel.shape[0]):
#     for j in range(sampel.shape[1]):
#         if im[i, j] > 90:
#             im[i, j] = 70

cv2.imshow('img', im)
cv2.waitKey(0)