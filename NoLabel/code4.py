# -*- coding: utf-8 -*-

#%%
import pandas as pd
import numpy as np
#%%
y_train_in = pd.read_pickle('Ytrain')
X_train_in = pd.read_pickle('Xtrain')
#%%
print(X_train_in.shape)
print(y_train_in.shape)

#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
some_digit = X_train_in[0]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image)
plt.axis("off")
plt.show()
#%%
from sklearn.model_selection import train_test_split
#X_train, X_test = train_test_split(X_train, test_size=0.2, random_state=42)
X_train, X_valid= X_train_in[:-5000], X_train_in[-5000:]
X_train_set = X_train
X_valid_set = X_valid

y_train, y_valid= y_train_in[:-5000], y_train_in[-5000:]
y_train_set = y_train
y_valid_set = y_valid
print(X_train_set.shape)
print(y_train_set.shape)

#%%
import tensorflow.keras as keras
conv_encoder = keras.models.Sequential([
        keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
        keras.layers.Conv2D(12, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(14, kernel_size=3, padding="SAME", activation="relu"),
])
conv_encoder.summary()
#%%
conv_decoder = keras.models.Sequential([
        keras.layers.Conv2DTranspose(12, kernel_size=3, strides=2, 
                                     padding="SAME", activation="relu",input_shape=[14, 14, 14]),
        keras.layers.Conv2DTranspose(1, kernel_size=3, strides=1, 
                                     padding="SAME", activation="sigmoid"),
        keras.layers.Reshape([28, 28])
])
conv_decoder.summary()

#%%
conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])
conv_ae.summary()
#%%
conv_ae.compile(loss="binary_crossentropy", optimizer="Adadelta",metrics=["binary_accuracy"])
history = conv_ae.fit(X_train_set, y_train_set, epochs=20, validation_data=(X_valid_set, y_valid_set))
#%%
conv_ae.save("model4.h5")
#%%
def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")

def show_reconstructions(model,  images=X_valid_set, n_images=20):
    reconstructions = model.predict(images[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))    
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(X_valid[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])
        
#%%
show_reconstructions(conv_ae) 
plt.show()
#%%
denoised_images = conv_ae.predict(X_valid_set)
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
n_image = 100
plt.figure(figsize=(10,10))
example_images = denoised_images[:n_image]
plot_digits(example_images, images_per_row=10)
plt.show()
#%%
plt.figure(figsize=(10,10))
example_images = y_valid_set[:n_image]
plot_digits(example_images, images_per_row=10)
plt.show()
#%%
plt.figure(figsize=(10,10))
example_images = X_valid_set[:n_image]
plot_digits(example_images, images_per_row=10)
plt.show()