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
df_all = df_all.append(df_xc1, ignore_index = True)

df_all = shuffler(df_all)

#Divide dataframe into x and y
df_x = pd.DataFrame(df_all.iloc[:, 0:115])
df_y = pd.DataFrame(df_all.iloc[:, 115])
df_pt2_x = pd.DataFrame(df_pt2.iloc[:, 0:115])
df_pt2_y = pd.DataFrame(df_pt2.iloc[:, 115])
df_xc_x = pd.DataFrame(df_xc1.iloc[:, 0:115])
df_xc_y = pd.DataFrame(df_xc1.iloc[:, 115])
df_xc2_x = pd.DataFrame(df_xc2.iloc[:, 0:115])
df_xc2_y = pd.DataFrame(df_xc2.iloc[:, 115])

#Normalize the x dataframe
df_x = normalize(df_x)
df_pt2 = normalize(df_pt2_x)
df_xc = normalize(df_xc_x)
df_xc2 = normalize(df_xc2_x)

#One-Hot encoding labels and transform into array
df_y = label_encoder(df_y)

# Divide dataset into training set and testing set
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.40)

class Net(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, h_dim)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.linear2 = torch.nn.Linear(h_dim, out_dim)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

train_x = torch.FloatTensor(train_x.values.astype(np.float32))
test_x = torch.tensor(test_x.values.astype(np.float32))
train_y = torch.tensor(train_y.astype(np.float32))
test_y = torch.tensor(test_y.astype(np.float32))
xc_x = torch.tensor(df_xc.values.astype(np.float32))
pt2_x = torch.tensor(df_pt2.values.astype(np.float32))

epochs = 500
input_dim = 115
output_dim = 2 #Number of clasees
h_dim = 100
lr_rate = 1e-6

model = Net(input_dim, h_dim, output_dim)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

n_test, ys = test_y.shape

ac_array = []
epoch_array = []
loss_array = []

def training(epochs, model, train_data, labels, vali_data, target):
    for e in range(int(epochs)):
        y_pred = model(train_data)
        # Compute and print loss
        loss = criterion(y_pred, labels)
        if e % 100 == 99:
            epoch_array.append(e)
            loss_f = float(loss)
            loss_array.append(loss_f)
            print(e, "Loss:", loss_f)
            total = n_test
            correct = 0.0
            outputs = model(vali_data)
            _b, pred = torch.max(outputs, 1)
            vb, label = torch.max(target, 1)
            correct+= float((pred == label).sum())
            accuracy = float(100*(correct/total))
            ac_array.append(accuracy)
            print('Accuracy: {:.4f}'.format(accuracy))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

training(epochs, model, train_x, train_y, test_x, test_y)

plot(epoch_array, loss_array, title='MSE Loss along epochs', yax='Loss')
plot(epoch_array, ac_array, title='Accuracy along epochs', yax='Accuracy')
