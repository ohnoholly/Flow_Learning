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
import matplotlib.pyplot as plt

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
    precision = true_pos/postive
    #print("P:", precision)
    recall = true_pos/truth
    #print("R:", recall)
    f_score = 2*(precision*recall)/(precision+recall)
    return f_score

def plot(x_axis, y_axis, y_axis2=None, label1='', label2=''):
    fig = plt.figure()
    fig.suptitle('MSE Loss along epochs', fontsize=14, fontweight='bold')
    plt.plot(x_axis, y_axis, label=label1)
    if y_axis2 != None:
        plt.plot(x_axis, y_axis2, label=label2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.show()

# Load all the data from the CSV file
BM_DATA_PATH = "../../../Dataset/Botnet_Detection/Philips_B120N10_Baby_Monitor"
DB_DATA_PATH = "../../../Dataset/Botnet_Detection/Danmini_Doorbell"
ET_DATA_PATH = "../../../Dataset/Botnet_Detection/Ecobee_Thermostat"
PT_DATA_PATH = "../../../Dataset/Botnet_Detection/PT_838_Security Camera"
XC_DATA_PATH = "../../../Dataset/Botnet_Detection/XCS7_1002_WHT_Security_Camera"
df_bm_b = pd.read_csv(BM_DATA_PATH+"/benign_traffic.csv")
df_bm_m = pd.read_csv(BM_DATA_PATH+"/Mirai/udp.csv")
df_db_b = pd.read_csv(DB_DATA_PATH+"/benign_traffic.csv")
df_db_m = pd.read_csv(DB_DATA_PATH+"/Mirai/udp.csv")
df_et_b = pd.read_csv(ET_DATA_PATH+"/benign_traffic.csv")
df_et_m = pd.read_csv(ET_DATA_PATH+"/Mirai/udp.csv")
df_pt_b = pd.read_csv(PT_DATA_PATH+"/benign_traffic.csv")
df_pt_m = pd.read_csv(PT_DATA_PATH+"/Mirai/udp.csv")
df_xc_b = pd.read_csv(XC_DATA_PATH+"/benign_traffic.csv")
df_xc_m = pd.read_csv(XC_DATA_PATH+"/Mirai/udp.csv")

#Assign the label to each dataframe
df_bm_b = df_bm_b.assign(label = 'b')
df_db_b = df_db_b.assign(label = 'b')
df_et_b = df_et_b.assign(label = 'b')
df_pt_b = df_pt_b.assign(label = 'b')
df_xc_b = df_xc_b.assign(label = 'b')
df_bm_m = df_bm_m.assign(label = 'm')
df_db_m = df_db_m.assign(label = 'm')
df_et_m = df_et_m.assign(label = 'm')
df_pt_m = df_pt_m.assign(label = 'm')
df_xc_m = df_xc_m.assign(label = 'm')

#Combine the benign traffic and malicious traffic
df_bm = df_bm_b
df_bm = df_bm.append(df_bm_m, ignore_index = True)
df_db = df_db_b
df_db = df_db.append(df_db_m, ignore_index = True)
df_et = df_et_b
df_et = df_et.append(df_et_m, ignore_index = True)
df_pt = df_pt_b
df_pt = df_pt.append(df_pt_m, ignore_index = True)
df_xc = df_xc_b
df_xc = df_xc.append(df_xc_m, ignore_index = True)

def shuffler(df):
  return df.reindex(np.random.permutation(df.index))

# Shuffle the rows in dataframe
df_bm = shuffler(df_bm)
df_db = shuffler(df_db)
df_et = shuffler(df_et)
df_pt = shuffler(df_pt)
df_xc = shuffler(df_xc)

# Create a dataset on server for initial model (second version)
df_server = pd.DataFrame()
df_server = df_server.append(df_et.sample(frac =.25), ignore_index=True)
df_server = df_server.append(df_pt.sample(frac =.25), ignore_index=True)
df_server = df_server.append(df_xc.sample(frac =.25), ignore_index=True)

#Divide dataframe into x and y
df_s_x = pd.DataFrame(df_server.iloc[:, 0:115])
df_s_y = pd.DataFrame(df_server.iloc[:, 115])
df_bm_x = pd.DataFrame(df_bm.iloc[:, 0:115])
df_bm_y = pd.DataFrame(df_bm.iloc[:, 115])
df_db_x = pd.DataFrame(df_db.iloc[:, 0:115])
df_db_y = pd.DataFrame(df_db.iloc[:, 115])

#Normalize the x dataframe
df_s_x = normalize(df_s_x)
df_bm_x = normalize(df_bm_x)
df_db_x = normalize(df_db_x)

#One-Hot encoding labels and transform into array
s_y = label_encoder(df_s_y)
bm_y = label_encoder(df_bm_y)
db_y = label_encoder(df_db_y)

#The different models ready to use:
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

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


def logistic_training(epochs, model, data, labels):
    for epochs in range(int(epochs)):
        optimizer.zero_grad() ## Zero out the gradient
        outputs = model(data) ## Call forward

        loss = ((outputs - labels)**2).sum() ## softmax
        if epochs % 10 == 9:
            print(loss)
        loss.backward() ## Accumulated gradient updates into x
        optimizer.step()

def net_training(epochs, model, data, labels ,loss_array, epoch_array):
    for e in range(int(epochs)):
        y_pred = model(data)

        # Compute and print loss
        loss = criterion(y_pred, labels)
        if e % 10 == 0:
            print(e, loss.data)
            loss_array.append(float(loss))
            epoch_array.append(e)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Create the virtual workers and hook them together
hook = sy.TorchHook(torch)

BM = sy.VirtualWorker(hook, id='BM')
DB = sy.VirtualWorker(hook, id='DB')

#Split the dataset itto training and testing
from sklearn.model_selection import train_test_split
b_train_x, b_test_x, b_train_y, b_test_y = train_test_split(df_bm_x, bm_y, test_size=0.90)
d_train_x, d_test_x, d_train_y, d_test_y = train_test_split(df_db_x, db_y, test_size=0.90)

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

n_bm, y_bm = t_b_test_y.shape
n_db, y_db = t_d_test_y.shape

print("n_bm", n_bm)
print("n_db", n_db)

b_x_train_ptr = t_b_train_x.send(BM)
b_x_test_ptr = t_b_test_x.send(BM)
b_y_train_ptr = t_b_train_y.send(BM)
b_y_test_ptr = t_b_test_y.send(BM)
d_x_train_ptr = t_d_train_x.send(DB)
d_x_test_ptr = t_d_test_x.send(DB)
d_y_train_ptr = t_d_train_y.send(DB)
d_y_test_ptr = t_d_test_y.send(DB)

#Set up parameters
epochs = 200
input_dim = 115
output_dim = 2 #Number of clasees
h_dim = 50
lr_rate = 1e-6


model = torch.nn.Module()
loss_array = []
epoch_array = []
#Get the input from the user
model_type = input("Enter the model to use: ")


if model_type == "0":
    model = LogisticRegression(input_dim, output_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)
    # Train the initial model on Server
    logistic_training(epochs, model, tensor_server_x, tensor_server_y)
elif model_type == "1":
    model = Net(input_dim, h_dim, output_dim)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)
    net_training(epochs, model, tensor_server_x, tensor_server_y, loss_array, epoch_array)
    plot(epoch_array, loss_array)


#Send the initial model to the workers
BM_model = model.copy().send(BM)
DB_model = model.copy().send(DB)

print(BM_model)

BM_opt = torch.optim.SGD(params=BM_model.parameters(),lr=1e-8)
DB_opt = torch.optim.SGD(params=DB_model.parameters(),lr=1e-8)
BM_loss_a = []
DB_loss_a = []
epoch_local_array = []
for e in range(200):

    #Baby Monitor
    #print(b_x_train_ptr.get())
    BM_pred = BM_model(b_x_train_ptr)

    # Compute and print loss
    BM_loss = criterion(BM_pred, b_y_train_ptr)

    # Zero gradients, perform a backward pass, and update the weights.
    BM_opt.zero_grad()
    BM_loss.backward()
    BM_opt.step()

    #Door Bell
    DB_pred = DB_model(d_x_train_ptr)

    # Compute and print loss
    DB_loss = criterion(DB_pred, d_y_train_ptr)

    # Zero gradients, perform a backward pass, and update the weights.
    DB_opt.zero_grad()
    DB_loss.backward()
    DB_opt.step()

    if e%10 == 0:
        BM_loss_f = float(BM_loss.get().data)
        DB_loss_f = float(DB_loss.get().data)
        print(e, "BM_loss:", BM_loss_f)
        print(e, "DB_loss:", DB_loss_f)
        BM_loss_a.append(BM_loss_f)
        DB_loss_a.append(DB_loss_f)
        epoch_local_array.append(e)
        total_b = n_bm
        correct = 0.0
        outputs_b = BM_model(b_x_test_ptr)
        _b, pred_b = torch.max(outputs_b.data, 1)
        vb, labels_b = torch.max(b_y_test_ptr.data, 1)
        correct+= float((pred_b == labels_b).sum())
        accuracy_b = float(100*(correct/total_b))
        fscore=f_score(pred_b, labels_b)
        print("Iteration:", e)
        print('BM Accuracy: {:.4f}'.format(accuracy_b), 'F1_score: ', fscore)



        total_d = n_db
        correct = 0.0
        outputs_d = DB_model(d_x_test_ptr)
        _d, pred_d = torch.max(outputs_d.data, 1)
        vd, labels_d = torch.max(d_y_test_ptr.data, 1)
        correct+= float((pred_d == labels_d).sum())
        accuracy_d = float(100*(correct/total_d))
        fscore=f_score(pred_d, labels_d)
        print('DB Accuracy: {:.4f}'.format(accuracy_d),'F1_score: ', fscore)

plot(epoch_local_array, BM_loss_a, DB_loss_a, 'BM', 'DB')
