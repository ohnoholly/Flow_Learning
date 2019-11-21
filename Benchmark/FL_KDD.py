import sys
import numpy as np
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
import syft as sy
import time

start_time = time.time()

data_path = "../../../Dataset/KDD99/kddcup99.csv"
dataset = pd.read_csv(data_path, sep=',', usecols=range(0, 42))
print("Dataset Shape:", dataset.shape)

data_server = dataset.sample(frac=0.5, random_state=1)
dataset = dataset.drop(data_server.index)
data_alice = dataset.sample(frac=0.5, random_state=1)
data_bob = dataset.drop(data_alice.index)

from sklearn import preprocessing

def encoding(data):
    for col in data.columns:
        if data[col].dtype == type(object):
            le_x = preprocessing.LabelEncoder()
            le_x.fit(data[col])
            data[col] = le_x.transform(data[col])
    return data

data_server_x = pd.DataFrame(data_server.iloc[:, 0:41])
data_server_y = pd.DataFrame(data_server.iloc[:, 41])
data_alice_x = pd.DataFrame(data_alice.iloc[:, 0:41])
data_alice_y = pd.DataFrame(data_alice.iloc[:, 41])
data_bob_x = pd.DataFrame(data_bob.iloc[:, 0:41])
data_bob_y = pd.DataFrame(data_bob.iloc[:, 41])

new_class = {'back':'abnormal', 'buffer_overflow':'abnormal', 'ftp_write':'abnormal', 'guess_passwd':'abnormal', 'imap':'abnormal',
            'ipsweep':'abnormal', 'land':'abnormal', 'loadmodule':'abnormal', 'multihop':'abnormal', 'neptune':'abnormal', 'nmap':'abnormal',
            'perl':'abnormal', 'phf':'abnormal', 'pod':'abnormal', 'portsweep':'abnormal', 'rootkit':'abnormal', 'satan':'abnormal',
            'smurf':'abnormal', 'spy':'abnormal', 'teardrop':'abnormal', 'warezclient':'abnormal', 'warezmaster':'abnormal'}
data_server_y = data_server_y.replace(new_class)
data_alice_y = data_alice_y.replace(new_class)
data_bob_y = data_bob_y.replace(new_class)

data_server_x = encoding(data_server_x)
data_server_y = encoding(data_server_y)
data_alice_x = encoding(data_alice_x)
data_alice_y = encoding(data_alice_y)
data_bob_x = encoding(data_bob_x)
data_bob_y = encoding(data_bob_y)

def normalize(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

data_server_x = normalize(data_server_x)
data_alice_x = normalize(data_alice_x)
data_bob_x = normalize(data_bob_x)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(data_server_y)
data_server_y = enc.transform(data_server_y).toarray() #Encode the classes to a binary array
enc.fit(data_alice_y)
data_alice_y = enc.transform(data_alice_y).toarray()
enc.fit(data_bob_y)
data_bob_y = enc.transform(data_bob_y).toarray()
print(data_server_y.shape)
print(data_alice_y.shape)
print(data_bob_y.shape)


hook = sy.TorchHook(torch)
Alice = sy.VirtualWorker(hook, id='Alice')
Bob = sy.VirtualWorker(hook, id='Bob')
from sklearn.model_selection import train_test_split
a_train_x, a_test_x, a_train_y, a_test_y = train_test_split(data_alice_x, data_alice_y, test_size=0.20)
b_train_x, b_test_x, b_train_y, b_test_y = train_test_split(data_bob_x, data_bob_y, test_size=0.20)

tensor_server_x = torch.tensor(data_server_x.values.astype(np.float32))
tensor_server_y = torch.tensor(data_server_y.astype(np.float32))
t_a_train_x = torch.tensor(a_train_x.values.astype(np.float32))
t_a_test_x = torch.tensor(a_test_x.values.astype(np.float32))
t_a_train_y = torch.tensor(a_train_y.astype(np.float32))
t_a_test_y = torch.tensor(a_test_y.astype(np.float32))
t_b_train_x = torch.tensor(b_train_x.values.astype(np.float32))
t_b_test_x = torch.tensor(b_test_x.values.astype(np.float32))
t_b_train_y = torch.tensor(b_train_y.astype(np.float32))
t_b_test_y = torch.tensor(b_test_y.astype(np.float32))
a_x_train_ptr = t_a_train_x.send(Alice)
a_x_test_ptr = t_a_test_x.send(Alice)
a_y_train_ptr = t_a_train_y.send(Alice)
a_y_test_ptr = t_a_test_y.send(Alice)
b_x_train_ptr = t_b_train_x.send(Bob)
b_x_test_ptr = t_b_test_x.send(Bob)
b_y_train_ptr = t_b_train_y.send(Bob)
b_y_test_ptr = t_b_test_y.send(Bob)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

#Setting up some parameters
epochs = 3
input_dim = 41
output_dim = 2 #Number of clasees
lr_rate = 0.0001

model = LogisticRegression(input_dim, output_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

def training(epochs, model, data, labels):
    print(epochs)
    for epochs in range(int(epochs)):
        optimizer.zero_grad() ## Zero out the gradient
        outputs = model(data) ## Call forward
        loss = ((outputs - labels)**2).sum() ## softmax
        loss.backward() ## Accumulated gradient updates into x
        print(loss)
        optimizer.step()

tensor_server_y = tensor_server_y.squeeze()
## Train the initial model on Server
print(tensor_server_y.shape)
training(epochs, model, tensor_server_x, tensor_server_y)

rtime_FL_init = time.time() - start_time
print("Running time for initial training:", rtime_FL_init)

bobs_model = model.copy().send(Bob)
alices_model = model.copy().send(Alice)

bobs_opt = torch.optim.SGD(params=bobs_model.parameters(),lr=lr_rate)
alices_opt = torch.optim.SGD(params=alices_model.parameters(),lr=lr_rate)

for i in range(2):

    # Train Bob's Model
    bobs_opt.zero_grad()
    bobs_pred = bobs_model(b_x_train_ptr)
    bobs_loss = ((bobs_pred - b_y_train_ptr)**2).sum()
    bobs_loss.backward()

    bobs_opt.step()
    bobs_loss = bobs_loss.get().data

    # Train Alice's Model
    alices_opt.zero_grad()
    alices_pred = alices_model(a_x_train_ptr)
    alices_loss = ((alices_pred - a_y_train_ptr)**2).sum()
    alices_loss.backward()

    alices_opt.step()
    alices_loss = alices_loss.get().data

    total = 24701
    correct = 0
    outputs_a = alices_model(a_x_test_ptr)
    _a, pred_a = torch.max(outputs_a.data, 1)
    va, labels_a = torch.max(a_y_test_ptr.data, 1)
    correct+= (pred_a == labels_a).sum()
    accuracy_a = 100*correct/total
    print("Iteration:", i, "ALice Accuracy: ", accuracy_a.get().data)

    correct = 0
    outputs_b = bobs_model(b_x_test_ptr)
    _b, pred_b = torch.max(outputs_b.data, 1)
    vb, labels_b = torch.max(b_y_test_ptr.data, 1)
    correct+= (pred_b == labels_b).sum()
    accuracy_b = 100*correct/total
    print("Iteration:", i, "Bob Accuracy: ", accuracy_b.get().data)


rtime = time.time() - start_time
rtime_FL_local = rtime-rtime_FL_init
print("Running time for secnodary training:", rtime_FL_local)
