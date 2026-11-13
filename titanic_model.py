#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 21:06:40 2022

@author: basti07
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.optimizers import SGD
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Daten aus CSV laden
data = pd.read_csv("datasets/train.csv", usecols=[1,2,4,5])
# print(data)

# Daten-Aufbereitung
for index, row in data.iterrows():
    data["Sex"] = np.where(data["Sex"]=="male", "0", "1")
    
data[['Age']] = data[['Age']].fillna(value=data[['Age']].mean())

dataset = data.values
dataset = dataset.astype(int)

X = dataset[:,1:4]
Y = dataset[:,0].reshape(891, 1)

X = normalize(X)

print(X[0:10])
print(X.shape)
print(Y[0:10])
print(Y.shape)


# Model erstellen
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Model trainieren
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, Y, batch_size=32, epochs=100, validation_split=0.33, verbose=1, shuffle=True)
model.save("model.h5")
predictions = model.evaluate(X, Y)
print("Loss: {}".format(predictions[0]))
print("Accuracy: {}".format(predictions[1]))
