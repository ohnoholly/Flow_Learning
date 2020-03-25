# Feature Importance with Extra Trees Classifier
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# load data
PT_DATA_PATH = "../../../Dataset/Botnet_Detection/PT_838_Security Camera"
#PT2_DATA_PATH = "../../../Dataset/Botnet_Detection/PT737E_Security Camera"
XC_DATA_PATH = "../../../Dataset/Botnet_Detection/XCS7_1002_WHT_Security_Camera"
#XC2_DATA_PATH = "../../../Dataset/Botnet_Detection/XCS7_1003_WHT_Security_Camera"
df_pt_1 = pd.read_csv(PT_DATA_PATH+"/benign_traffic.csv")
#df_pt_2 = pd.read_csv(PT2_DATA_PATH+"/benign_traffic.csv")
df_xc_1 =  pd.read_csv(XC_DATA_PATH+"/benign_traffic.csv")
#df_xc_2 =  pd.read_csv(XC2_DATA_PATH+"/benign_traffic.csv")

df_pt1 = df_pt_1.assign(label = 'pt1')
#df_pt2 = df_pt_2.assign(label = 'pt2')
df_xc1 = df_xc_1.assign(label = 'xc1')
#df_xc2 = df_xc_2.assign(label = 'xc2')

df_all = df_pt1
df_all = df_all.append(df_xc1, ignore_index = True)

array = df_all.values
X = array[:,0:115]
Y = array[:,115]
# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, Y)
print(model.feature_importances_)

# feature extraction
model2 = LogisticRegression(solver='lbfgs')
rfe = RFE(model2, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

# feature extraction
pca = PCA(n_components=50)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)
