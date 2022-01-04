# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:39:31 2021

@author: tamphi
"""
#%%
import pandas as pd
import numpy as np
import pickle

#%%

y_train = pd.read_pickle('project3trainlabel.pkl')
X_train = pd.read_pickle('project3trainset.pkl')

#%%
print(X_train.shape)
print(y_train.shape)
#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
some_digit = X_train[0]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image)
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
#Split into train and test set

from sklearn.model_selection import train_test_split
X_train_set, X_test_set = train_test_split(X_train, test_size=0.1, random_state=42)
y_train_set, y_test_set = train_test_split(y_train, test_size=0.1, random_state=42)

X_train_set, X_valid_set= X_train_set[:-5000], X_train_set[-5000:]
y_train_set, y_valid_set = y_train_set[:-5000], y_train_set[-5000:]
X_mean = X_train_set.mean(axis=0, keepdims=True)
X_std = X_train_set.std(axis=0, keepdims=True) + 1e-7
X_train_set = (X_train_set - X_mean) / X_std
X_valid_set = (X_valid_set - X_mean) / X_std
X_test_set = (X_test_set - X_mean) / X_std

#%%
print(X_train_set.shape)
print(X_valid_set.shape)
print(X_test_set.shape)
print(y_train_set.shape)
print(y_valid_set.shape)
print(y_test_set.shape)
#%%

X_train_set = X_train_set[..., np.newaxis]
X_valid_set = X_valid_set[..., np.newaxis]
X_test_set = X_test_set[..., np.newaxis]

#%%
import tensorflow.keras as keras
from functools import partial
from keras.wrappers.scikit_learn import KerasClassifier
#%%
def create_model(activation = 'relu',
                 optimizer = "nadam",
                 dropout_rate = 0.2,
                 kernel =3,
                 ):
    DefaultConv2D = partial(keras.layers.Conv2D,
    kernel_size=kernel, activation=activation, padding="SAME")
    model_grid = keras.models.Sequential([
        DefaultConv2D(filters=28, kernel_size=7, input_shape=[28, 28, 1]),
        keras.layers.MaxPooling2D(pool_size=3),
        DefaultConv2D(filters=32),
        DefaultConv2D(filters=32),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=36),
        DefaultConv2D(filters=36),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=36, activation= activation),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(units=32, activation= activation),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(units=10, activation= 'softmax'),
])
    model_grid.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model_grid

#%%
from sklearn.model_selection import GridSearchCV

activation =  ['relu', 'sigmoid'] # softmax, softplus, softsign 
dropout_rate = [ 0.2, 0.3, 0.4]
optimizers = ['Adadelta', 'Nadam']
epochs = [20,30] # add 50, 100, 150 etc
param_grid = dict(activation=activation,optimizer = optimizers,dropout_rate = dropout_rate, epochs=epochs)

#%%
model_grid = KerasClassifier(build_fn=create_model, verbose=0)
grid = GridSearchCV(estimator=model_grid, param_grid=param_grid, cv=5, verbose = 3)
grid_result = grid.fit(X_train_set, y_train_set)
#%%
print(grid_result.best_params_)
print(grid_result.best_score_)
#%%

DefaultConv2D = partial(keras.layers.Conv2D,
kernel_size=3, activation='relu', padding="SAME")
model = keras.models.Sequential([
        DefaultConv2D(filters=28, kernel_size=7, input_shape=[28, 28, 1]),
        keras.layers.MaxPooling2D(pool_size=3),
        DefaultConv2D(filters=32),
        DefaultConv2D(filters=32),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=36),
        DefaultConv2D(filters=36),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=36, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(units=10, activation='softmax'),
])
    
#%%
model.summary()
#%%
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adadelta", metrics=["accuracy"])
history = model.fit(X_train_set, y_train_set, epochs=20, validation_data=(X_valid_set, y_valid_set))
score = model.evaluate(X_test_set, y_test_set)
X_new = X_test_set# pretend we have new images
y_pred = model.predict(X_new)
#%%
model.save("model3.h5")
#%%
y_train_pred  = model.predict_classes(X_train_set)
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
f1_score = [0]*10
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
print("  N   Precision            Recall               F1-Score")
for i in range(10):
    recall_arr[i] = tp_arr[i] / (tp_arr[i]+fn_arr[i])
    precision_arr[i] = tp_arr[i] / (tp_arr[i]+fp_arr[i])
    f1_score[i] = 2* (precision_arr[i]*recall_arr[i])/(precision_arr[i] + recall_arr[i])
    print(" ",i," ",precision_arr[i]," ",recall_arr[i],"  ",f1_score[i])
