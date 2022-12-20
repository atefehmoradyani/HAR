import numpy as np
import pandas as pd
import os
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import sys

data = []
cat = np.zeros((50, 50))

for r in range(50):
    cat[r, r] = 1


data_train = []
data_test = []
cat_train = []
cat_test = []

lst2 = os.listdir('features2/')
lst = os.listdir('features/')
# lst = os.listdir('conv/')
# i = 0
cat_file = []
cat_file2 = []
# jj = 0
for file in lst2:
    print(file)
    name = file.split('.')
    name = name[0]
    number = int(name)
    for file2 in lst:
        name2 = file2.split('.')
        name2 = name2[0]
        number2 = int(name2)
        if number2 == number:
            conv3d = []
            conv3d = np.array(conv3d)
            # for i in range(fi.shape[1]):
            #     conv3d = np.append(conv3d, sum(fi[:, i]))

            ############### conv3d1 + conv3d2
            # input_data = np.zeros((1, 5000))
            # fe = pd.read_csv('conv/' + file, header=None)
            # fi = np.array(fe)
            # for i in range(fi.shape[1]):
            #     conv3d = np.append(conv3d, sum(fi[:, i]))
            # conv3d = np.reshape(conv3d, (1, 1000))
            # fi = np.reshape(fi, (1, len(fi) * 1000))
            # input_data[0, 0: fi.shape[1]] = fi
            # fi = np.append(input_data, conv3d)
            # data.append(fi)

            ############### DAE 1
            # fe = pd.read_csv('features/' + file, header=0)
            # fi = np.array(fe)
            # fi = np.reshape(fi, (1, len(fi) * 1000))
            # input_data = np.zeros((1, 5000))
            # input_data[0, 0: fi.shape[1]] = fi
            # data.append(input_data)

            ############### DAE on DAE
            # fe = pd.read_csv('features2/' + file, header=0)
            # fi = np.array(fe)
            # data.append(fi)

            ############### conv3d on conv3d
            # fe = pd.read_csv('conv/' + file, header=None)
            # fi = np.array(fe)
            # for i in range(fi.shape[1]):
            #     conv3d = np.append(conv3d, sum(fi[:, i]))
            # data.append(conv3d)

            ############### DAE2 + DAE1
            fe2 = pd.read_csv('features2/' + file, header=0)
            fe = pd.read_csv('features/' + file2, header=0)
            fe = np.array(fe)
            fe2 = np.array(fe2)
            fe2 = np.reshape(fe2, (1, 1000))
            fe = np.reshape(fe, (1, len(fe) * 1000))
            input_data = np.zeros((1, 5000))
            input_data[0, 0: fe.shape[1]] = fe
            input_data = np.append(input_data, fe2)
            data.append(input_data)
            # if(number == 10010):
            #     data = np.array(data)
            #     print(data.shape)

            ############### DAE2 + onv3d on DAE1
            # fe2 = pd.read_csv('features2/' + file2, header=0)
            # fe = pd.read_csv('features/' + file, header=0)
            # fi = np.array(fe)
            # for i in range(fi.shape[1]):
            #     conv3d = np.append(conv3d, sum(fi[:, i]))
            # fe2 = np.array(fe2)
            # conv3d = np.append(conv3d, fe2)
            # data.append(conv3d)

            ############### con3d on DAE1
            # fe = pd.read_csv('features/' + file, header=0)
            # fi = np.array(fe)
            # for i in range(1000):
            #     conv3d = np.append(conv3d, sum(fi[:, i]))
            # data.append(conv3d)


            # fi = np.reshape(fi, (1, len(fi) * 1000))
            # input_data[0, 0: fi.shape[1]] = fi
            # fe2 = np.array(fe2)
            # input_data = np.append(input_data, fe2)
            # input_data = np.append(input_data, conv3d)
            # data.append(input_data)
            # data.append(conv3d)
            # data.append(fe2)


            if 1000 < number < 2000:
                cat_file.append(list(cat[0, :]))
                cat_file2.append(0)
            elif 2000 < number < 3000:
                cat_file.append(list(cat[1, :]))
                cat_file2.append(1)
            elif 3000 < number < 4000:
                cat_file.append(list(cat[2, :]))
                cat_file2.append(2)
            elif 4000 < number < 5000:
                cat_file.append(list(cat[3, :]))
                cat_file2.append(3)
            elif 5000 < number < 6000:
                cat_file.append(list(cat[4, :]))
                cat_file2.append(4)
            elif 6000 < number < 7000:
                cat_file.append(list(cat[5, :]))
                cat_file2.append(5)
            elif 7000 < number < 8000:
                cat_file.append(list(cat[6, :]))
                cat_file2.append(6)
            elif 8000 < number < 9000:
                cat_file.append(list(cat[7, :]))
                cat_file2.append(7)
            elif 9000 < number < 10000:
                cat_file.append(list(cat[8, :]))
                cat_file2.append(8)
            elif 10000 < number < 11000:
                cat_file.append(list(cat[9, :]))
                cat_file2.append(9)
            elif 11000 < number < 12000:
                cat_file.append(list(cat[10, :]))
                cat_file2.append(10)
            elif 12000 < number < 13000:
                cat_file.append(list(cat[11, :]))
                cat_file2.append(11)
            elif 13000 < number < 14000:
                cat_file.append(list(cat[12, :]))
                cat_file2.append(12)
            elif 14000 < number < 15000:
                cat_file.append(list(cat[13, :]))
                cat_file2.append(13)
            elif 15000 < number < 16000:
                cat_file.append(list(cat[14, :]))
                cat_file2.append(14)
            elif 16000 < number < 17000:
                cat_file.append(list(cat[15, :]))
                cat_file2.append(15)
            elif 17000 < number < 18000:
                cat_file.append(list(cat[16, :]))
                cat_file2.append(16)
            elif 18000 < number < 19000:
                cat_file.append(list(cat[17, :]))
                cat_file2.append(17)
            elif 19000 < number < 20000:
                cat_file.append(list(cat[18, :]))
                cat_file2.append(18)
            elif 20000 < number < 21000:
                cat_file.append(list(cat[19, :]))
                cat_file2.append(19)
            elif 21000 < number < 22000:
                cat_file.append(list(cat[20, :]))
                cat_file2.append(20)
            elif 22000 < number < 23000:
                cat_file.append(list(cat[21, :]))
                cat_file2.append(21)
            elif 23000 < number < 24000:
                cat_file.append(list(cat[22, :]))
                cat_file2.append(22)
            elif 24000 < number < 25000:
                cat_file.append(list(cat[23, :]))
                cat_file2.append(23)
            elif 25000 < number < 26000:
                cat_file.append(list(cat[24, :]))
                cat_file2.append(24)
            elif 26000 < number < 27000:
                cat_file.append(list(cat[25, :]))
                cat_file2.append(25)
            elif 27000 < number < 28000:
                cat_file.append(list(cat[26, :]))
                cat_file2.append(26)
            elif 28000 < number < 29000:
                cat_file.append(list(cat[27, :]))
                cat_file2.append(27)
            elif 29000 < number < 30000:
                cat_file.append(list(cat[28, :]))
                cat_file2.append(28)
            elif 30000 < number < 31000:
                cat_file.append(list(cat[29, :]))
                cat_file2.append(29)
            elif 31000 < number < 32000:
                cat_file.append(list(cat[30, :]))
                cat_file2.append(30)
            elif 32000 < number < 33000:
                cat_file.append(list(cat[31, :]))
                cat_file2.append(31)
            elif 33000 < number < 34000:
                cat_file.append(list(cat[32, :]))
                cat_file2.append(32)
            elif 34000 < number < 35000:
                cat_file.append(list(cat[33, :]))
                cat_file2.append(33)
            elif 35000 < number < 36000:
                cat_file.append(list(cat[34, :]))
                cat_file2.append(34)
            elif 36000 < number < 37000:
                cat_file.append(list(cat[35, :]))
                cat_file2.append(35)
            elif 37000 < number < 38000:
                cat_file.append(list(cat[36, :]))
                cat_file2.append(36)
            elif 38000 < number < 39000:
                cat_file.append(list(cat[37, :]))
                cat_file2.append(37)
            elif 39000 < number < 40000:
                cat_file.append(list(cat[38, :]))
                cat_file2.append(38)
            elif 40000 < number < 41000:
                cat_file.append(list(cat[39, :]))
                cat_file2.append(39)
            elif 41000 < number < 42000:
                cat_file.append(list(cat[40, :]))
                cat_file2.append(40)
            elif 42000 < number < 43000:
                cat_file.append(list(cat[41, :]))
                cat_file2.append(41)
            elif 43000 < number < 44000:
                cat_file.append(list(cat[42, :]))
                cat_file2.append(42)
            elif 44000 < number < 45000:
                cat_file.append(list(cat[43, :]))
                cat_file2.append(43)
            elif 45000 < number < 46000:
                cat_file.append(list(cat[44, :]))
                cat_file2.append(44)
            elif 46000 < number < 47000:
                cat_file.append(list(cat[45, :]))
                cat_file2.append(45)
            elif 47000 < number < 48000:
                cat_file.append(list(cat[46, :]))
                cat_file2.append(46)
            elif 48000 < number < 49000:
                cat_file.append(list(cat[47, :]))
                cat_file2.append(47)
            elif 49000 < number < 50000:
                cat_file.append(list(cat[48, :]))
                cat_file2.append(48)
            elif 50000 < number:
                cat_file.append(list(cat[49, :]))
                cat_file2.append(49)
            break


cat_file = np.array(cat_file)
data = np.array(data)
# data = data[:, 0, :]
a, b = cat_file.shape
# data = data[:, 0, :]

# print("a = {} , b = {}".format(a, b))

print("data shape = ", data.shape)
#train and test data

# data = preprocessing.normalize(data, norm='l1')

# for item in cat:
#     myclass = []
#     for i in range(a):
#         idx = 0
#         if np.array_equal(item, cat_file[i]):
#             # if idx == 100:
#             #     break
#             myclass.append(data[i])
#             # idx = idx + 1
#     print(len(myclass))
#     for j in range(round(len(myclass)*0.90)):
#         data_train.append(myclass[j])
#         cat_train.append(item)
#     for j in range(round(len(myclass)*0.90), len(myclass)):
#         data_test.append(myclass[j])
#         cat_test.append(item)
for item in range(50):
    myclass = []
    for i in range(a):
        # idx = 0
        if item == cat_file2[i]:
            # if idx == 100:
            #     break
            myclass.append(data[i])
            # idx = idx + 1
    for j in range(round(len(myclass)*0.80)):
        data_train.append(myclass[j])
        cat_train.append(item)
    for j in range(round(len(myclass)*0.80), len(myclass)):
        data_test.append(myclass[j])
        cat_test.append(item)
data_test = np.array(data_test)
data_train = np.array(data_train)
cat_test = np.array(cat_test)
cat_train = np.array(cat_train)

# x, y = datasets.load_iris(return_X_y=True)
# print(x)
# print(y)
model = OneVsRestClassifier(SVC(max_iter=30000, probability=True))
model.fit(data_train, cat_train)
np.set_printoptions(threshold=np.inf)
print(model.predict(data_test))
print('=======================')
print(cat_test)
print("------------probability")
print(model.predict_proba(data_test))
print("--------- mean accuracy")
print(model.score(data_test, cat_test))
