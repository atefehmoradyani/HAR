from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

hmdb = pd.read_csv('confusion/hmdb.csv', delimiter=",", header=None)
hmdb = np.array(hmdb)
print(confusion_matrix(hmdb[:, 0], hmdb[:, 1]))
x = confusion_matrix(hmdb[:, 0], hmdb[:, 1])
# print(sns.heatmap(x/np.sum(x), annot=True,
#             fmt='.2%', cmap='Blues'))

classifier = KNeighborsClassifier()
classifier.fit(hmdb, hmdb[:, 1])
# plot_confusion_matrix(classifier, hmdb[:, 0], hmdb[:, 1])
print("-----------")
print(accuracy_score(hmdb[:, 0], hmdb[:, 1]))
print("============")
print(recall_score(hmdb[:, 0], hmdb[:, 1], average=None))
# print(precision_score(hmdb[:, 0], hmdb[:, 1]))

