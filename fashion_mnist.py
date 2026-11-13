#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 17:30:38 2021

@author: basti07
"""

import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import gzip
import matplotlib.pyplot as plt
from pathlib import Path
from random import randint
from tensorflow.keras.callbacks import TensorBoard

def load_fashion_data():
    # Trainingsdaten
    ## Labels
    with gzip.open("data/fashion/train-labels-idx1-ubyte.gz") as path:
        train_labels = np.frombuffer(path.read(), np.uint8, offset=8)
        
    ## Bilder
    with gzip.open("data/fashion/train-images-idx3-ubyte.gz") as path:
        train_images = np.frombuffer(path.read(), np.uint8, offset=16).reshape(len(train_labels), 28,28)/255.0
        
    # Testdaten
    ## Labels
    with gzip.open("data/fashion/t10k-labels-idx1-ubyte.gz") as path:
        test_labels = np.frombuffer(path.read(), np.uint8, offset=8)
        
    ## Bilder
    with gzip.open("data/fashion/t10k-images-idx3-ubyte.gz") as path:
        test_images = np.frombuffer(path.read(), np.uint8, offset=16).reshape(len(test_labels),28,28)/255.0
        
    test_images = np.reshape(test_images, (-1, 28, 28, 1))
    train_images = np.reshape(train_images, (-1, 28, 28, 1))
    
    seed = 33
    
    # Shuffel der Trainingsdaten
    train_images, train_labels = shuffle(train_images, train_labels, random_state=seed)
    
    # Shuffel der Testdaten
    test_images, test_labels = shuffle(test_images, test_labels, random_state=seed)
    
    # OneHotEncoding
    train_labels = tf.keras.utils.to_categorical(train_labels,10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)
    
    return train_images, train_labels, test_images, test_labels


def get_test_image(index):
    return np.reshape(test_images[index],(-1,28,28,1))

# Laden der Dateien
train_images, train_labels, test_images, test_labels = load_fashion_data()

plt.imshow(train_images[30])
print(train_labels[30])

class_labels = {
    0 : "T-Shirt",
    1 : "Trouser",
    2 : "Pullover",
    3 : "Dress",
    4 : "Coat",
    5 : "Sandal",
    6 : "Shirt",
    7 : "Sneaker",
    8 : "Bag",
    9 : "Ankel boot"
}


# Bilder Testen/Anzeigen
def show_test_images(index):
    plt.title(class_labels[np.argmax(test_labels[index])])
    plt.imshow(np.reshape(test_images[index], (28,28)), cmap="Greys")
    plt.show()

# show_test_images(9999)    


# Modell erstellen
def fashion_model():
    fash_model = tf.keras.models.Sequential()
    
    # Kodierungsblock
    ## Convolutional Layer 1
    fash_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[5,5], activation="relu"))
    fash_model.add(tf.keras.layers.MaxPooling2D(pool_size=[2,2]))
    
    ## Convolutional Layer 2
    fash_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[5,5], activation="relu"))
    fash_model.add(tf.keras.layers.MaxPooling2D(pool_size=[2,2]))
    
    # Prädikationsblock
    fash_model.add(tf.keras.layers.Flatten())
    fash_model.add(tf.keras.layers.Dense(1024))
    fash_model.add(tf.keras.layers.Dropout(0.5)) 
    fash_model.add(tf.keras.layers.Dense(10, activation="softmax"))
    
    return fash_model


# Callback
tf_callback = TensorBoard(log_dir="logs", histogram_freq=1)

# Modell trainieren
model_path = Path("models/fashion_mnist.h5")

if not model_path.is_file():
    model = fashion_model()
    model.compile(optimizer="adam", loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])  
    model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.1, verbose=True, callbacks=[tf_callback])
    model.save("models/fashion_mnist.h5")
else:
    model = tf.keras.models.load_model("models/fashion_mnist.h5")

# Modell evaluieren
ergebnis = model.evaluate(test_images, test_labels)
print("Loss: {}".format(ergebnis[0]))
print("Accuracy: {}".format(ergebnis[1]))


# Predictions
pred_int = randint(0, 999)
predictions = model.predict(get_test_image(pred_int))
print(predictions)
show_test_images(pred_int)
print(class_labels[np.argmax(predictions)])