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

def plot(x_axis, y_axis, y_axis2=None, label1='', label2='', title=''):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.plot(x_axis, y_axis, label=label1)
    if y_axis2 != None:
        plt.plot(x_axis, y_axis2, label=label2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
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


# Shuffle the rows in dataframe
df_pt1 = shuffler(df_pt1)
df_pt2 = shuffler(df_pt2)
df_xc1 = shuffler(df_xc1)
df_xc2 = shuffler(df_xc2)

# Create a dataset on server for initial model (second version)
df_server = pd.DataFrame()
df_server = df_server.append(df_pt1.sample(frac =.25), ignore_index=True)
df_server = df_server.append(df_pt2.sample(frac =.25), ignore_index=True)
df_server = df_server.append(df_xc1.sample(frac =.25), ignore_index=True)
#df_server = df_server.append(df_xc2.sample(frac =.25), ignore_index=True)

# Create two dataframes for each device mixed two external classes
df_pt = pd.DataFrame()
df_pt = df_pt.append(df_pt1)
df_pt = df_pt.append(df_pt2.sample(frac =.25), ignore_index = True)
df_pt = df_pt.append(df_xc2.sample(frac =.25), ignore_index = True)
df_xc = pd.DataFrame()
df_xc = df_xc.append(df_xc1)
df_xc = df_xc.append(df_xc2.sample(frac =.25), ignore_index = True)
df_xc = df_xc.append(df_pt2.sample(frac =.25), ignore_index = True)

#Divide dataframe into x and y
df_s_x = pd.DataFrame(df_server.iloc[:, 0:115])
df_s_y = pd.DataFrame(df_server.iloc[:, 115])
df_pt_x = pd.DataFrame(df_pt.iloc[:, 0:115])
df_pt_y = pd.DataFrame(df_pt.iloc[:, 115])
df_xc_x = pd.DataFrame(df_xc.iloc[:, 0:115])
df_xc_y = pd.DataFrame(df_xc.iloc[:, 115])

##Normalize the x dataframe
df_s_x = normalize(df_s_x)
df_pt_x = normalize(df_pt_x)
df_xc_x = normalize(df_xc_x)

print(df_s_x.shape)
print(df_pt_x.shape)
print(df_xc_x.shape)

#One-Hot encoding labels and transform into array
s_y = label_encoder(df_s_y)
pt_y = label_encoder(df_pt_y)
xc_y = label_encoder(df_xc_y)

#The different models ready to use:
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

class Net(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, h_dim)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.linear2 = torch.nn.Linear(h_dim, out_dim)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def logistic_training(epochs, model, data, labels, loss_array, epoch_array):
    for epochs in range(int(epochs)):
        optimizer.zero_grad() ## Zero out the gradient
        outputs = model(data) ## Call forward

        loss = ((outputs - labels)**2).sum() ## softmax
        if epochs % 10 == 0:
            print(loss.data)
            loss_array.append(float(loss))
            epoch_array.append(epochs)
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

PT = sy.VirtualWorker(hook, id='PT')
XC = sy.VirtualWorker(hook, id='XC')

#Split the dataset itto training and testing
from sklearn.model_selection import train_test_split
p_train_x, p_test_x, p_train_y, p_test_y = train_test_split(df_pt_x, pt_y, test_size=0.20)
x_train_x, x_test_x, x_train_y, x_test_y = train_test_split(df_xc_x, xc_y, test_size=0.20)

tensor_server_x = torch.FloatTensor(df_s_x.values.astype(np.float32))
tensor_server_y = torch.FloatTensor(s_y.astype(np.float32))
t_p_train_x = torch.FloatTensor(p_train_x.values.astype(np.float32))
t_p_test_x = torch.tensor(p_test_x.values.astype(np.float32))
t_p_train_y = torch.tensor(p_train_y.astype(np.float32))
t_p_test_y = torch.tensor(p_test_y.astype(np.float32))
t_x_train_x = torch.tensor(x_train_x.values.astype(np.float32))
t_x_test_x = torch.tensor(x_test_x.values.astype(np.float32))
t_x_train_y = torch.tensor(x_train_y.astype(np.float32))
t_x_test_y = torch.tensor(x_test_y.astype(np.float32))

n_pt, y_pt = t_p_test_y.shape
n_xc, y_xc = t_x_test_y.shape

print("n_pt", n_pt)
print("n_xc", n_xc)

p_x_train_ptr = t_p_train_x.send(PT)
p_x_test_ptr = t_p_test_x.send(PT)
p_y_train_ptr = t_p_train_y.send(PT)
p_y_test_ptr = t_p_test_y.send(PT)
x_x_train_ptr = t_x_train_x.send(XC)
x_x_test_ptr = t_x_test_x.send(XC)
x_y_train_ptr = t_x_train_y.send(XC)
x_y_test_ptr = t_x_test_y.send(XC)

#Set up parameters
epochs = 200
input_dim = 115
output_dim = 3 #Number of clasees
h_dim = 50
lr_rate = 1e-8


model = nn.Module()
loss_array = []
epoch_array = []
#Get the input from the user
model_type = input("Enter the model to use: ")
start_time = time.time()

if model_type == "0":
    model = LogisticRegression(input_dim, output_dim)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)
    # Train the initial model on Server
    logistic_training(epochs, model, tensor_server_x, tensor_server_y, loss_array, epoch_array)
    rtime_FL_init = time.time() - start_time
    print("Running time for initial training:", rtime_FL_init)
    plot(epoch_array, loss_array, title='MSE Loss along epochs')
elif model_type == "1":
    model = Net(input_dim, h_dim, output_dim)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)
    net_training(epochs, model, tensor_server_x, tensor_server_y, loss_array, epoch_array)
    rtime_FL_init = time.time() - start_time
    print("Running time for initial training:", rtime_FL_init)
    plot(epoch_array, loss_array, title='MSE Loss along epochs')


#Send the initial model to the workers
model1 = model.copy().send(PT)
model2 = model.copy().send(XC)


opt1 = torch.optim.SGD(params=model1.parameters(),lr=1e-6)
opt2 = torch.optim.SGD(params=model2.parameters(),lr=1e-5)
loss_a1 = []
loss_a2 = []
ac1 = []
ac2 = []
f1 = []
f2 = []
epoch_local_array = []
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


rtime = time.time() - start_time
rtime_FL_local = rtime-rtime_FL_init
print("Running time for secnodary training:", rtime_FL_local)
plot(epoch_local_array, loss_a1, loss_a2, 'PT', 'XC', title='MSE Loss along epochs')
plot(epoch_local_array, ac1, ac2, 'PT', 'XC', title='Accuracy along epochs')
plot(epoch_local_array, f1, f2, 'PT', 'XC', 'F1 score along epochs')
