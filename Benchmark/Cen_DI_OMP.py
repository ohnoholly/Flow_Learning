import sys
import numpy as np
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import syft as sy
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time

start_time = time.time()

def normalize(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def encoding(data):
    for col in data.columns:
        if data[col].dtype == type(object):
            le_x = preprocessing.LabelEncoder()
            le_x.fit(data[col])
            data[col] = le_x.transform(data[col])
    return data

from sklearn.preprocessing import OneHotEncoder
def label_encoder(df):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df)
    print(enc.categories_)
    df_array = enc.transform(df).toarray() #Encode the classes to a binary array
    return df_array

def f_score(pred, label):
    pred = torch.unsqueeze(pred, 0)
    label = torch.unsqueeze(label, 1)
    true_pos = torch.mm(pred, label)
    true_pos = float(true_pos)
    #print("true_pos", true_pos)
    postive = float(pred.sum())
    #print(postive)
    truth = float(label.sum())
    #print(truth)
    precision = true_pos/(postive+0.00001)
    #print("P:", precision)
    recall = true_pos/truth
    #print("R:", recall)
    f_score = 2*(precision*recall)/(precision+recall+0.00001)
    return f_score

def plot(x_axis, y_axis, y_axis2=None, label1='', label2='', title='', yax=''):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.plot(x_axis, y_axis, label=label1)
    if y_axis2 != None:
        plt.plot(x_axis, y_axis2, label=label2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(yax, fontsize=12)
    plt.legend()
    plt.show()

def group_hist(labels, v1, v2, v3=None):
    x = np.arange(len(labels))  # the label locations
    width = 0.36  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/3, v1, width/3, label='size=100')
    rects2 = ax.bar(x, v2, width/3, label='size=200')
    rects3 = ax.bar(x+ width/3, v3, width/3, label='size=300')

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per sparse level')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()


def shuffler(df):
  return df.reindex(np.random.permutation(df.index))

# Load all the data from the CSV file
PT_DATA_PATH = "../../../Dataset/Botnet_Detection/PT_838_Security Camera"
PT2_DATA_PATH = "../../../Dataset/Botnet_Detection/PT737E_Security Camera"
XC_DATA_PATH = "../../../Dataset/Botnet_Detection/XCS7_1002_WHT_Security_Camera"
XC2_DATA_PATH = "../../../Dataset/Botnet_Detection/XCS7_1003_WHT_Security_Camera"
df_pt_1 = pd.read_csv(PT_DATA_PATH+"/benign_traffic.csv")
df_pt_2 = pd.read_csv(PT2_DATA_PATH+"/benign_traffic.csv")
df_xc_1 =  pd.read_csv(XC_DATA_PATH+"/benign_traffic.csv")
df_xc_2 =  pd.read_csv(XC2_DATA_PATH+"/benign_traffic.csv")

#Assign the label to each dataframe
df_pt1 = df_pt_1.assign(label = 'pt1')
df_pt2 = df_pt_2.assign(label = 'pt2')
df_xc1 = df_xc_1.assign(label = 'xc1')
df_xc2 = df_xc_2.assign(label = 'xc2')

df_all = df_pt1
df_all = shuffler(df_all)

#Sample some instances from the dataset
test_set = df_pt1.sample(frac=0.001, random_state=1)
test_set = test_set.append(df_pt2.sample(frac=0.001, random_state=1))
test_set = shuffler(test_set)
print(test_set.shape)

train_x = pd.DataFrame(df_all.iloc[:, 0:115])
train_y = pd.DataFrame(df_all.iloc[:, 115])
test_x = pd.DataFrame(test_set.iloc[:, 0:115])
test_y = pd.DataFrame(test_set.iloc[:, 115])

train_x = normalize(train_x)
test_x = normalize(test_x)

train_y = label_encoder(train_y)
test_y = label_encoder(test_y)

train_x = torch.FloatTensor(train_x.values.astype(np.float32))
test_x = torch.tensor(test_x.values.astype(np.float32))
train_y = torch.tensor(train_y.astype(np.float32))
test_y = torch.tensor(test_y.astype(np.float32))

def maxind(V, r):
    res=[]
    for row in V:
        l2 = np.linalg.norm(row)
        row = row.unsqueeze(1)
        row = torch.transpose(row,0,1)
        row = row.squeeze(0)
        r = r.squeeze(0)
        inner = float(torch.dot(row, r))
        s = float(inner/l2)
        res.append(s)
        res_tensor = torch.FloatTensor(res)

    _, i = torch.max(res_tensor, 0)
    return i


## Note: The vector(row) picked from the matrix cannot be re-sized directly.
## y is expected as 2D matrix
def OMP(y, V, sl, ep):
    c_encode = torch.zeros([1, len(V)], dtype=torch.float32)
    s_index = []
    r = y
    k = 1
    while (k <= sl) and (np.linalg.norm(r) >= ep):
        i = int(maxind(V, r))
        s_index.append(i)
        temp = []
        for sk in s_index:
            temp_v = V[sk]
            if len(temp)==0:
                temp=temp_v
                temp = temp.unsqueeze(0)
            else:
                temp_v = V[sk].unsqueeze(0)
                temp = torch.cat((temp, temp_v),0)


        s_matrix = temp
        s_matrix = s_matrix.type(torch.FloatTensor)
        s_i = torch.pinverse(s_matrix)
        c = torch.mm(y, s_i)
        r = y - torch.mm(c, s_matrix)
        k = k+1

    #print(s_index)
    v=0
    for sk in s_index:
        c_encode[0][sk] = c[0][v]
        v = v+1
    return c_encode

def projecting(train, test, size, sp, ep, threshold):
    re = torch.zeros([len(test), 2], dtype=torch.float32)
    for ids, vector in enumerate(test):
        v = vector.unsqueeze(0)
        c = OMP(v, train[:size][:], sp, ep)
        t = v - torch.mm(c, train[:size][:])
        distance = np.linalg.norm(t)
        #print("id:", ids, "dist:",distance)
        if distance < threshold:
            re[ids][0] = 1.0
        else:
            re[ids][1] = 1.0
    #print(re)
    return re

sp_array = []
size1 = []
size2 = []
size3 = []
sp = 5
while sp <= 20:
    size = 100
    while size <= 300:
        re = projecting(train_x, test_x, size, sp, 0.001, 100000000000.0)
        total = len(test_x)
        correct = 0.0
        _, pred = torch.max(re, 1)
        v, label = torch.max(test_y, 1)
        correct+= float((pred == label).sum())
        accuracy = float(100*(correct/total))
        print("Sp:", sp, "Size:", size, 'Accuracy: {:.4f}'.format(accuracy))
        if size == 100:
            size1.append(accuracy)
        elif size == 200:
            size2.append(accuracy)
        elif size == 300:
            size3.append(accuracy)

        size = size + 100
    sp_array.append(sp)
    sp = sp + 5

group_hist(sp_array, size1, size2, size3)
