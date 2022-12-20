import os
import numpy as np

lst = os.listdir('features2')
# for file in lst:
#     print(file)
#     name = file.split('.')
#     name = name[0]
#     number = int(name)
#     if 1000 < number < 2000:
#         cat_file.append(list(cat[0, :]))
#     elif 2000 < number < 3000:
#         cat_file.append(list(cat[1, :]))
#     elif 3000 < number < 4000:
#         cat_file.append(list(cat[2, :]))
#     elif 4000 < number < 5000:
#         cat_file.append(list(cat[3, :]))
#     elif 5000 < number < 6000:
#         cat_file.append(list(cat[4, :]))
#     elif 6000 < number < 7000:
#         cat_file.append(list(cat[5, :]))
#     elif 7000 < number < 8000:
#         cat_file.append(list(cat[6, :]))
#     elif 8000 < number < 9000:
#         cat_file.append(list(cat[7, :]))
#     elif 9000 < number < 10000:
#         cat_file.append(list(cat[8, :]))
#     elif 10000 < number < 11000:
#         cat_file.append(list(cat[9, :]))
#     elif 11000 < number < 12000:
#         cat_file.append(list(cat[10, :]))
#     elif 12000 < number < 13000:
#         cat_file.append(list(cat[11, :]))
#     elif 13000 < number < 14000:
#         cat_file.append(list(cat[12, :]))
#     elif 14000 < number < 15000:
#         cat_file.append(list(cat[13, :]))
#     elif 15000 < number < 16000:
#         cat_file.append(list(cat[14, :]))
#     elif 16000 < number < 17000:
#         cat_file.append(list(cat[15, :]))
#     elif 17000 < number < 18000:
#         cat_file.append(list(cat[16, :]))
#     elif 18000 < number < 19000:
#         cat_file.append(list(cat[17, :]))
#     elif 19000 < number < 20000:
#         cat_file.append(list(cat[18, :]))
#     elif 20000 < number < 21000:
#         cat_file.append(list(cat[19, :]))
#     elif 21000 < number < 22000:
#         cat_file.append(list(cat[20, :]))
#     elif 22000 < number < 23000:
#         cat_file.append(list(cat[21, :]))
#     elif 23000 < number < 24000:
#         cat_file.append(list(cat[22, :]))
#     elif 24000 < number < 25000:
#         cat_file.append(list(cat[23, :]))
#     elif 25000 < number < 26000:
#         cat_file.append(list(cat[24, :]))
#     elif 26000 < number < 27000:
#         cat_file.append(list(cat[25, :]))
#     elif 27000 < number < 28000:
#         cat_file.append(list(cat[26, :]))
#     elif 28000 < number < 29000:
#         cat_file.append(list(cat[27, :]))
#     elif 29000 < number < 30000:
#         cat_file.append(list(cat[28, :]))
#     elif 30000 < number < 31000:
#         cat_file.append(list(cat[29, :]))
#     elif 31000 < number < 32000:
#         cat_file.append(list(cat[30, :]))
#     elif 32000 < number < 33000:
#         cat_file.append(list(cat[31, :]))
#     elif 33000 < number < 34000:
#         cat_file.append(list(cat[32, :]))
#     elif 34000 < number < 35000:
#         cat_file.append(list(cat[33, :]))
#     elif 35000 < number < 36000:
#         cat_file.append(list(cat[34, :]))
#     elif 36000 < number < 37000:
#         cat_file.append(list(cat[35, :]))
#     elif 37000 < number < 38000:
#         cat_file.append(list(cat[36, :]))
#     elif 38000 < number < 39000:
#         cat_file.append(list(cat[37, :]))
#     elif 39000 < number < 40000:
#         cat_file.append(list(cat[38, :]))
#     elif 40000 < number < 41000:
#         cat_file.append(list(cat[39, :]))
#     elif 41000 < number < 42000:
#         cat_file.append(list(cat[40, :]))
#     elif 42000 < number < 43000:
#         cat_file.append(list(cat[41, :]))
#     elif 43000 < number < 44000:
#         cat_file.append(list(cat[42, :]))
#     elif 44000 < number < 45000:
#         cat_file.append(list(cat[43, :]))
#     elif 45000 < number < 46000:
#         cat_file.append(list(cat[44, :]))
#     elif 46000 < number < 47000:
#         cat_file.append(list(cat[45, :]))
#     elif 47000 < number < 48000:
#         cat_file.append(list(cat[46, :]))
#     elif 48000 < number < 49000:
#         cat_file.append(list(cat[47, :]))
#     elif 49000 < number < 50000:
#         cat_file.append(list(cat[48, :]))
#     elif 50000 < number:
#         cat_file.append(list(cat[49, :]))
