import time
import pandas as pd
import numpy as np
import matplotlib as plot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

start_time = time.time()

#Load the data from the CSV file
data_path = "../../../Dataset/kddcup99.csv"
dataset = pd.read_csv(data_path, sep=',', usecols=range(0, 42))
print("Dataset Shape:", dataset.shape)

#Seprerate the dataset for the features and the labels
input_x = dataset.iloc[:, 0:41]
input_y = dataset.iloc[:, 41]

#Split the dataset into training and testing set
train_x, test_x, train_y, test_y = train_test_split(input_x, input_y, test_size=0.20)

# Re-assign the label to only two classes
new_class = {'back':'abnormal', 'buffer_overflow':'abnormal', 'ftp_write':'abnormal', 'guess_passwd':'abnormal', 'imap':'abnormal',
            'ipsweep':'abnormal', 'land':'abnormal', 'loadmodule':'abnormal', 'multihop':'abnormal', 'neptune':'abnormal', 'nmap':'abnormal',
            'perl':'abnormal', 'phf':'abnormal', 'pod':'abnormal', 'portsweep':'abnormal', 'rootkit':'abnormal', 'satan':'abnormal',
            'smurf':'abnormal', 'spy':'abnormal', 'teardrop':'abnormal', 'warezclient':'abnormal', 'warezmaster':'abnormal'}
train_y = train_y.replace(new_class)
test_y = test_y.replace(new_class)

#Encode the label and features to the representative number
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

#Start training process
from sklearn.svm import LinearSVC
clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(train_x, train_y)
print("Running Time:", (time.time()-start_time))

# Start predicting process
prid = clf.predict(test_x)

#Evaluate the accuracy and the confussion matrix
print("Accuracy:", clf.score(test_x, test_y))


print(confusion_matrix(prid, test_y))
print("Running Time:", (time.time()-start_time))
