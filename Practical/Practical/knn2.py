import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df=pd.read_csv("C:/Users/DELL/Desktop/diabetes.csv")
print(df.head())

print(df.shape)
print(df.columns)
print(df.isna().sum())

x=df.drop(["Outcome"], axis=1)
y=df["Outcome"]

print(x.shape)
print(y.shape)

x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size=0.2, random_state=200)

knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

metrics.accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()

Accuracy= (tn+tp)/(tn+tp+fp+fn)
print(Accuracy)