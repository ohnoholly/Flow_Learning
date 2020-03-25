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

from sklearn.preprocessing import OneHotEncoder
def label_encoder(df):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df)
    print(enc.categories_)
    df_array = enc.transform(df).toarray() #Encode the classes to a binary array
    return df_array

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

# Load all the data from the CSV file
PT_DATA_PATH = "../../../Dataset/Botnet_Detection/PT_838_Security Camera"
XC_DATA_PATH = "../../../Dataset/Botnet_Detection/XCS7_1002_WHT_Security_Camera"

df_pt_b = pd.read_csv(PT_DATA_PATH+"/benign_traffic.csv")
df_pt_m = pd.read_csv(PT_DATA_PATH+"/Mirai/ack.csv")
df_xc_b =  pd.read_csv(XC_DATA_PATH+"/benign_traffic.csv")
df_xc_m =  pd.read_csv(XC_DATA_PATH+"/Mirai/ack.csv")

#Assign the label to each dataframe
df_ptb = df_pt_b.assign(label = 'b')
df_ptm = df_pt_m.assign(label = 'm')
df_xcb = df_xc_b.assign(label = 'b')
df_xcm = df_xc_m.assign(label = 'm')


df_ptb = df_ptb.sample(n=1000, random_state=1)
df_ptm = df_ptm.sample(n=100, random_state=1)
df_xcb = df_xcb.sample(n=1000, random_state=1)
df_xcm = df_xcm.sample(n=100, random_state=1)


df_pt = df_ptb
df_xc = df_xcb
df_pt = df_pt.append(df_ptm)
df_xc = df_xc.append(df_xcm)



#Divide dataframe into x and y
df_ptx = pd.DataFrame(df_pt.iloc[:, 0:115])
df_pty = pd.DataFrame(df_pt.iloc[:, 115])
df_xcx = pd.DataFrame(df_xc.iloc[:, 0:115])
df_xcy = pd.DataFrame(df_xc.iloc[:, 115])


#Normalize the x dataframe
df_ptx = normalize(df_ptx)
df_xcx = normalize(df_xcx)


#One-Hot encoding labels and transform into array
df_pty = label_encoder(df_pty)
df_xcy = label_encoder(df_xcy)

# Divide dataset into training set and testing set
from sklearn.model_selection import train_test_split
train_ptx, test_ptx, train_pty, test_pty = train_test_split(df_ptx, df_pty, test_size=0.40)
train_xcx, test_xcx, train_xcy, test_xcy = train_test_split(df_xcx, df_pty, test_size=0.40)

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

train_px = torch.FloatTensor(train_ptx.values.astype(np.float32))
test_px = torch.tensor(test_ptx.values.astype(np.float32))
train_py = torch.tensor(train_pty.astype(np.float32))
test_py = torch.tensor(test_pty.astype(np.float32))
train_xx = torch.FloatTensor(train_xcx.values.astype(np.float32))
test_xx = torch.tensor(test_xcx.values.astype(np.float32))
train_xy = torch.tensor(train_xcy.astype(np.float32))
test_xy = torch.tensor(test_xcy.astype(np.float32))

# Create the virtual workers and hook them together
hook = sy.TorchHook(torch)

PT = sy.VirtualWorker(hook, id='PT')
XC = sy.VirtualWorker(hook, id='XC')


p_x_train_ptr = train_px.send(PT)
p_x_test_ptr = test_px.send(PT)
p_y_train_ptr = train_py.send(PT)
p_y_test_ptr = test_py.send(PT)
x_x_train_ptr = train_xx.send(XC)
x_x_test_ptr = test_xx.send(XC)
x_y_train_ptr = train_xy.send(XC)
x_y_test_ptr = test_xy.send(XC)

epochs = 500
input_dim = 115
output_dim = 2 #Number of clasees
h_dim = 50

model = Net(input_dim, h_dim, output_dim)
criterion = torch.nn.MSELoss(reduction='sum')

model1 = model.copy().send(PT)
model2 = model.copy().send(XC)


opt1 = torch.optim.SGD(params=model1.parameters(),lr=1e-6)
opt2 = torch.optim.SGD(params=model2.parameters(),lr=1e-6)
loss_a1 = []
loss_a2 = []
ac1 = []
ac2 = []
f1 = []
f2 = []
epoch_local_array = []

n_pt, y_pt = test_pty.shape
n_xc, y_xc = test_xcy.shape

for e in range(500):

    #Clinet1
    pred1 = model1(p_x_train_ptr)

    # Compute and print loss
    loss1 = criterion(pred1, p_y_train_ptr)

    # Zero gradients, perform a backward pass, and update the weights.
    opt1.zero_grad()
    loss1.backward()
    opt1.step()

    #CLient2
    pred2 = model2(x_x_train_ptr)

    # Compute and print loss
    loss2 = criterion(pred2, x_y_train_ptr)

    # Zero gradients, perform a backward pass, and update the weights.
    opt2.zero_grad()
    loss2.backward()
    opt2.step()

    if e%50 == 0:
        loss_f1 = float(loss1.get().data)
        loss_f2 = float(loss2.get().data)
        print(e, "PT_loss:", loss_f1)
        print(e, "XC_loss:", loss_f2)
        loss_a1.append(loss_f1)
        loss_a2.append(loss_f2)
        epoch_local_array.append(e)
        total_p = n_pt
        correct = 0.0
        outputs1 = model1(p_x_test_ptr)
        _p, pred_p = torch.max(outputs1.data, 1)
        vp, labels_p = torch.max(p_y_test_ptr.data, 1)
        correct+= float((pred_p == labels_p).sum())
        accuracy_p = float(100*(correct/total_p))
        ac1.append(accuracy_p)
        fscore=f_score(pred_p, labels_p)
        f1.append(fscore)
        print("Iteration:", e)
        print('PT Accuracy: {:.4f}'.format(accuracy_p), 'F1_score: ', fscore)



        total_x = n_xc
        correct = 0.0
        outputs2 = model2(x_x_test_ptr)
        _x, pred_x = torch.max(outputs2.data, 1)
        vx, labels_x = torch.max(x_y_test_ptr.data, 1)
        correct+= float((pred_x == labels_x).sum())
        accuracy_x = float(100*(correct/total_x))
        ac2.append(accuracy_x)
        fscore=f_score(pred_x, labels_x)
        f2.append(fscore)
        print('XC Accuracy: {:.4f}'.format(accuracy_x),'F1_score: ', fscore)


plot(epoch_local_array, loss_a1, loss_a2, 'PT', 'XC', title='MSE Loss along epochs')
plot(epoch_local_array, ac1, ac2, 'PT', 'XC', title='Accuracy along epochs')
plot(epoch_local_array, f1, f2, 'PT', 'XC', 'F1 score along epochs')
