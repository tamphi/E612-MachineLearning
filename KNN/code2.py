# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 07:53:43 2021

@author: tamphi
"""
#%%
import pandas as pd
import numpy as np
import pickle

#%%
X_train = pd.read_pickle('mnist_X_train.pkl')
y_train = pd.read_pickle('mnist_y_train.pkl')
#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
some_digit = X_train[0]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary)
plt.axis("off")
plt.show()
#%%

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
#%%
plt.figure(figsize=(9,9))
example_images = X_train[:100]
plot_digits(example_images, images_per_row=10)
plt.show()

#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
#%%
from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline

non_linear_svm_clf = Pipeline([
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="poly",degree=4,coef0=1,C=2))
])
non_lin_model = non_linear_svm_clf.fit(X_train_scaled, y_train)
#%%
from sklearn.model_selection import cross_val_score
non_lin_svm_predict_score = cross_val_score(non_linear_svm_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
print(non_lin_svm_predict_score)

#%%
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 6)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier(n_neighbors=12,weights ='distance')
knn_clf.fit(X_train, y_multilabel)

from sklearn.model_selection import cross_val_score
knn_predict_score = cross_val_score(knn_clf, X_train, y_multilabel, cv=3, scoring="accuracy")
print(knn_predict_score)

#%%
#logistic reg
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
softmax_clf = Pipeline([
("scaler",StandardScaler()),
("softmax_clf", LogisticRegression(
  multi_class="multinomial",
  solver="lbfgs", C=5, random_state=42,max_iter=60000))
])
softmax_clf.fit(X_train_scaled,y_train)

from sklearn.model_selection import cross_val_score
logreg_predict_score = cross_val_score(softmax_clf, X_train_scaled, y_train, cv=10, scoring="accuracy")
print(logreg_predict_score)


#%%
from sklearn.model_selection import train_test_split
X_train_set, X_test_set = train_test_split(X_train, test_size=0.2, random_state=42)
y_train_set, y_test_set = train_test_split(y_train, test_size=0.2, random_state=42)

#%%
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('lr', softmax_clf), ('knn', knn_clf), ('svc', non_linear_svm_clf)],
    voting='hard')
voting_clf.fit(X_train_set, y_train_set)

from sklearn.metrics import accuracy_score
for clf in (softmax_clf, knn_clf, non_linear_svm_clf, voting_clf):
    clf.fit(X_train_set, y_train_set)
    y_pred = clf.predict(X_test_set)
    print(clf.__class__.__name__, accuracy_score(y_test_set, y_pred))


#%%
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(voting_clf, X_train_set, y_train_set, cv=10)
#%%
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_train_set, y_train_pred)
print(conf_matrix)
#%%
plt.matshow(conf_matrix, cmap=plt.cm.gray)
plt.show() 
#%%
row_sums = conf_matrix.sum(axis=1, keepdims=True)
norm_conf_mx = conf_matrix/ row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
#%%
import numpy as np
tp_arr = [0]*10
fp_arr = [0]*10
fn_arr = [0]*10

recall_arr = [0]*10
precision_arr = [0]*10
for i in range(10):
    rowsum =0
    colsum =[0]*10
    for j in range(10):
        rowsum = sum(conf_matrix[i])
        colsum = np.sum(conf_matrix,axis=0)
        if (i == j):
            fp_arr[i] = rowsum - conf_matrix[i][j]
            tp_arr[i] = conf_matrix[i][j]
            fn_arr[i] = colsum[i] - conf_matrix[i][j]
print("\n")
print("N   Precision            Recall")
for i in range(10):
    recall_arr[i] = tp_arr[i] / (tp_arr[i]+fn_arr[i])
    precision_arr[i] = tp_arr[i] / (tp_arr[i]+fp_arr[i])
    print(i," ",precision_arr[i]," ",recall_arr[i])

#%%
        