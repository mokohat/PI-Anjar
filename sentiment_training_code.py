train_path= './test_100k.csv'
#test_path= 'drive/My Drive/Sentiment/test_50k.csv'

import csv
import numpy as np
import math
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import regex as re
import string
from nltk.tag import CRFTagger
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import time
start = time.time()
data_train=pd.read_csv(train_path)
#data_test=pd.read_csv(test_path)
print(data_train.columns.tolist())

label_train= data_train['Sentiment'].to_numpy()
fitur_train=data_train['Clean_Tweet'].tolist()


from sklearn.feature_extraction.text import TfidfVectorizer
#tf_con=fitur_train.fit_transform(corpus)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect=TfidfVectorizer()
vect.fit(fitur_train)
bag=vect.fit_transform(fitur_train)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(data_train, label_train, random_state=0)

from sklearn.svm import LinearSVC

svc=LinearSVC()
# svc.fit(x_train,y_train)

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test=train_test_split(data_train['Pre_Tweet'], label_train, random_state=0)
# data_train_index = x_train[:,0]
# data_train = x_train[:,1:]
# data_test_index = x_test[:,0]
# data_test = x_test[:,1:]
# svc = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
# clf_output = clf.fit(data_train, targets_train)

x_train, x_test, y_train, y_test=train_test_split(bag.todense(), label_train,test_size=0.25 ,random_state=42)

svc=LinearSVC()
# x_train=vect.fit_transform(x_train)
# x_test=vect.transform(x_test)
# matrix=x_train.todense()
# fit= [[np.mean(sublist) for sublist in matrix]]
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
# print("Accuracy:",svc.score(y_test, y_pred))

from sklearn.metrics import confusion_matrix

tn,fp,fn,tp=confusion_matrix(y_test,y_pred).ravel()
accuracy=(tn+tp)/(tn+tp+fn+fp)
precision=tp/(fp+tp)
recalls=tp/(fn+tp)
print("Accuracy " + "{:.2f}".format(accuracy*100) + "%")
# print(accuracy)
# print("precision")
print("Precision " + "{:.2f}".format(precision*100) + "%")
# print(precision)
print("Recalls " + "{:.2f}".format(recalls*100) + "%")
# print(recalls)
end = time.time()
print("Execution time " + "{:.2f}".format(end - start) + " seconds")