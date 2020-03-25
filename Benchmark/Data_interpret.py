import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns



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


x1 = df_pt_1.loc[:, 'MI_dir_L5_weight']
x2 = df_pt_2.loc[:, 'MI_dir_L5_weight']
x3 = df_xc_1.loc[:, 'MI_dir_L5_weight']
x4 = df_xc_2.loc[:, 'MI_dir_L5_weight']


# plot
kwargs = dict(alpha=0.5, bins=50)
plt.hist(x1, **kwargs, color='g', label='PT1')
plt.hist(x2, **kwargs, color='y', label='PT2')
plt.hist(x3, **kwargs, color='b', label='XC1')
plt.hist(x4, **kwargs, color='r', label='XC2')
plt.gca().set(title='Attribute Distribution', ylabel='Frequency', xlabel='Values')
plt.legend()
plt.show()
