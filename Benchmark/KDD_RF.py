import time
import pandas as pd
import numpy as np
import matplotlib as plot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

start_time = time.time()

data_path = "../../../Dataset/kddcup99.csv"
dataset = pd.read_csv(data_path, sep=',', usecols=range(0, 42))
print("Dataset Shape:", dataset.shape)

input_x = dataset.iloc[:, 0:41]
input_y = dataset.iloc[:, 41]

train_x, test_x, train_y, test_y = train_test_split(input_x, input_y, test_size=0.20)

new_class = {'back':'abnormal', 'buffer_overflow':'abnormal', 'ftp_write':'abnormal', 'guess_passwd':'abnormal', 'imap':'abnormal',
            'ipsweep':'abnormal', 'land':'abnormal', 'loadmodule':'abnormal', 'multihop':'abnormal', 'neptune':'abnormal', 'nmap':'abnormal',
            'perl':'abnormal', 'phf':'abnormal', 'pod':'abnormal', 'portsweep':'abnormal', 'rootkit':'abnormal', 'satan':'abnormal',
            'smurf':'abnormal', 'spy':'abnormal', 'teardrop':'abnormal', 'warezclient':'abnormal', 'warezmaster':'abnormal'}
train_y = train_y.replace(new_class)
test_y = test_y.replace(new_class)

le_y = preprocessing.LabelEncoder()
le_y.fit(train_y)
train_y = le_y.transform(train_y)
test_y = le_y.transform(test_y)

for col in train_x.columns:
    if train_x[col].dtype == type(object):
        le_x = preprocessing.LabelEncoder()
        le_x.fit(train_x[col])
        train_x[col] = le_x.transform(train_x[col])

for col in test_x.columns:
    if test_x[col].dtype == type(object):
        le_x = preprocessing.LabelEncoder()
        le_x.fit(test_x[col])
        test_x[col] = le_x.transform(test_x[col])

clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_x, train_y)

prid = clf.predict(test_x)
print(clf.feature_importances_)

print("Accuracy:", clf.score(test_x, test_y))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(prid, test_y))
print("Running Time:", (time.time()-start_time))
