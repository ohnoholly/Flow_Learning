import sys
import numpy as np

import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
import Sklearn_PyTorch
import syft as sy
from sklearn import preprocessing

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

# Load all the data from the CSV file
BM_DATA_PATH = "../../../Dataset/Botnet_Detection/Philips_B120N10_Baby_Monitor"
DB_DATA_PATH = "../../../Dataset/Botnet_Detection/Danmini_Doorbell"
ET_DATA_PATH = "../../../Dataset/Botnet_Detection/Ecobee_Thermostat"
df_bm_b = pd.read_csv(BM_DATA_PATH+"/benign_traffic.csv")
df_bm_m = pd.read_csv(BM_DATA_PATH+"/Mirai/udp.csv")
df_db_b = pd.read_csv(DB_DATA_PATH+"/benign_traffic.csv")
df_db_m = pd.read_csv(DB_DATA_PATH+"/Mirai/udp.csv")
df_et_b = pd.read_csv(ET_DATA_PATH+"/benign_traffic.csv")
df_et_m = pd.read_csv(ET_DATA_PATH+"/Mirai/udp.csv")

#Assign the label to each dataframe
df_bm_b = df_bm_b.assign(label = 'b')
df_db_b = df_db_b.assign(label = 'b')
df_et_b = df_et_b.assign(label = 'b')
df_bm_m = df_bm_m.assign(label = 'm')
df_db_m = df_db_m.assign(label = 'm')
df_et_m = df_et_m.assign(label = 'm')

#Combine the benign traffic and malicious traffic
df_bm = df_bm_b
df_bm = df_bm.append(df_bm_m, ignore_index = True)
df_db = df_db_b
df_db = df_db.append(df_db_m, ignore_index = True)
df_et = df_et_b
df_et = df_et.append(df_et_m, ignore_index = True)

def shuffler(df):
  return df.reindex(np.random.permutation(df.index))

# Shuffle the rows in dataframe
df_bm = shuffler(df_bm)
df_db = shuffler(df_db)
df_et = shuffler(df_et)

# Create a dataset on server for initial model
df_server = pd.DataFrame()
df_server = df_server.append(df_bm.sample(frac =.25), ignore_index=True)
df_server = df_server.append(df_db.sample(frac =.25), ignore_index=True)
df_server = df_server.append(df_et.sample(frac =.25), ignore_index=True)

#Divide dataframe into x and y
df_s_x = pd.DataFrame(df_server.iloc[:, 0:115])
df_s_y = pd.DataFrame(df_server.iloc[:, 115])
df_bm_x = pd.DataFrame(df_bm.iloc[:, 0:115])
df_bm_y = pd.DataFrame(df_bm.iloc[:, 115])
df_db_x = pd.DataFrame(df_db.iloc[:, 0:115])
df_db_y = pd.DataFrame(df_db.iloc[:, 115])
df_et_x = pd.DataFrame(df_et.iloc[:, 0:115])
df_et_y = pd.DataFrame(df_et.iloc[:, 115])

#Normalize the x dataframe
df_s_x = normalize(df_s_x)
df_bm_x = normalize(df_bm_x)
df_db_x = normalize(df_db_x)
df_et_x = normalize(df_et_x)

#One-Hot encoding labels and transform into array
s_y = label_encoder(df_s_y)
bm_y = label_encoder(df_bm_y)
db_y = label_encoder(df_db_y)
et_y = label_encoder(df_et_y)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

hook = sy.TorchHook(torch)

BM = sy.VirtualWorker(hook, id='BM')
DB = sy.VirtualWorker(hook, id='DB')

from sklearn.model_selection import train_test_split
b_train_x, b_test_x, b_train_y, b_test_y = train_test_split(df_bm_x, bm_y, test_size=0.20)
d_train_x, d_test_x, d_train_y, d_test_y = train_test_split(df_db_x, db_y, test_size=0.20)

tensor_server_x = torch.FloatTensor(df_s_x.values.astype(np.float32))
tensor_server_y = torch.FloatTensor(s_y.astype(np.float32))
t_b_train_x = torch.FloatTensor(b_train_x.values.astype(np.float32))
t_b_test_x = torch.tensor(b_test_x.values.astype(np.float32))
t_b_train_y = torch.tensor(b_train_y.astype(np.float32))
t_b_test_y = torch.tensor(b_test_y.astype(np.float32))
t_d_train_x = torch.tensor(d_train_x.values.astype(np.float32))
t_d_test_x = torch.tensor(d_test_x.values.astype(np.float32))
t_d_train_y = torch.tensor(d_train_y.astype(np.float32))
t_d_test_y = torch.tensor(d_test_y.astype(np.float32))

b_x_train_ptr = t_b_train_x.send(BM)
b_x_test_ptr = t_b_test_x.send(BM)
b_y_train_ptr = t_b_train_y.send(BM)
b_y_test_ptr = t_b_test_y.send(BM)
d_x_train_ptr = t_d_train_x.send(DB)
d_x_test_ptr = t_d_test_x.send(DB)
d_y_train_ptr = t_d_train_y.send(DB)
d_y_test_ptr = t_d_test_y.send(DB)

epochs = 3
input_dim = 115
output_dim = 2 #Number of clasees
lr_rate = 0.001

model = LogisticRegression(input_dim, output_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

def training(epochs, model, data, labels):
    print(epochs)
    for epochs in range(int(epochs)):
        print("In the loop")
        optimizer.zero_grad() ## Zero out the gradient
        outputs = model(data) ## Call forward
        print(outputs)
        print(labels)
        loss = ((outputs - labels)**2).sum() ## softmax
        print(loss)
        loss.backward() ## Accumulated gradient updates into x
        optimizer.step()

tensor_server_y = tensor_server_y.squeeze()
# Train the initial model on Server
training(epochs, model, tensor_server_x, tensor_server_y)

BM_model = model.copy().send(BM)
DB_model = model.copy().send(DB)

BM_opt = torch.optim.SGD(params=BM_model.parameters(),lr=lr_rate)
DB_opt = torch.optim.SGD(params=DB_model.parameters(),lr=lr_rate)


for i in range(2):

    # Train Bob's Model
    BM_opt.zero_grad()
    BM_pred = BM_model(b_x_train_ptr)
    BM_loss = ((BM_pred - b_y_train_ptr)**2).sum()
    BM_loss.backward()

    BM_opt.step()
    BM_loss = BM_loss.get().data

    # Train Alice's Model
    DB_opt.zero_grad()
    DB_pred = DB_model(d_x_train_ptr)
    DB_loss = ((DB_pred - d_y_train_ptr)**2).sum()
    DB_loss.backward()

    DB_opt.step()
    DB_loss = DB_loss.get().data

    total_b = 78455
    correct = 0
    outputs_b = BM_model(b_x_test_ptr)
    _b, pred_b = torch.max(outputs_b.data, 1)
    vb, labels_b = torch.max(b_y_test_ptr.data, 1)
    correct+= (pred_b == labels_b).sum()
    accuracy_b = 100*correct/total_b
    print("Iteration:", i, "BM Accuracy: ", accuracy_b.get().data)

    total_d = 57443
    correct = 0
    outputs_d = DB_model(d_x_test_ptr)
    _d, pred_d = torch.max(outputs_d.data, 1)
    vd, labels_d = torch.max(d_y_test_ptr.data, 1)
    correct+= (pred_d == labels_d).sum()
    accuracy_d = 100*correct/total_d
    print("Iteration:", i, "DB Accuracy: ", accuracy_d.get().data)
