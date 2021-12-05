
import numpy as np
import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score,  roc_curve, auc, classification_report
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt





#Read data
df = pd.read_csv("train-io.txt", sep=' ')

#Data Cleaning 
df.isnull().sum()
df=df.dropna()
df.head()
df.columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6','V7', 'V8', 'V9','V10', 'Class']


#Train/Test split¶
cleaned_df = df.copy()
y = df.pop('Class')
X = df
X1 = X.to_numpy()
Y1 = y.to_numpy()
train_X, val_X, train_y, val_y = train_test_split(X1, Y1, test_size = 0.2, random_state= 2)


scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
val_X = scaler.transform(val_X)
clf = KMeans(n_clusters=2, random_state=0).fit(train_X)
y_pred_train = clf.predict(train_X)
y_pred_test = clf.predict(val_X)
print(classification_report(val_y,y_pred_test))
print(accuracy_score(val_y, y_pred_test))
print('confusion matrix:\n', confusion_matrix(val_y, y_pred_test))
fpr, tpr, threshold = roc_curve(val_y, y_pred_test)
roc = auc(fpr, tpr)
plt.plot(fpr,tpr,color="yellow",label="ROC curve" % roc)
plt.plot([0, 1], [0, 1], color="black", linestyle="dotted")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.title(" Receiver operating characteristic for MLP")
plt.legend(loc="lower right")
plt.show()


df_test = pd.read_csv("test-i.txt", sep = " ", header = None)
test_pred = clf.predict(df_test)
test_pred = pd.DataFrame(test_pred)
test_pred
test_pred.to_csv("test-o.txt", index = False)


