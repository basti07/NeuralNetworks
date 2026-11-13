#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 22:24:07 2021

@author: basti07
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from pprint import pprint


#Datei Vorbereitung
iris_data = np.loadtxt("iris.csv", delimiter=",", dtype=str, skiprows=1)
labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
label_index = 0

for label in labels:
    print(label)
    iris_data[np.where(iris_data[:,4]==label), 4] = label_index
    label_index = label_index + 1
    
iris_data = iris_data.astype("float32")

input_data = iris_data[:,0:4]
output_data = iris_data[:,4].reshape(-1, 1)
output_data = tf.keras.utils.to_categorical(output_data, 3) 

iris_train_input, iris_test_input, iris_train_output, iris_test_output = train_test_split(input_data, output_data, test_size=0.2)


#Modell erstellen

csv_logger = CSVLogger("iris_classification_log.csv", separator=",", append=False)
tensorboard_log = TensorBoard(log_dir="logs/")

iris_model = Sequential()
#Mit input_dim(4) oder input_shape=(4,)
iris_model.add(Dense(5, input_dim=4, activation='tanh'))
iris_model.add(Dense(24, activation="relu"))
iris_model.add(Dense(3, activation="softmax"))
iris_model.summary()


#Modell trainieren
sgd = SGD(learning_rate=0.001)
iris_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
iris_model.fit(x=iris_train_input, y=iris_train_output, batch_size=10, epochs=500, verbose=1, callbacks=[csv_logger, tensorboard_log])


#Evulation des Modells
evalutation_result = iris_model.evaluate(iris_test_input, iris_test_output)

print("Loss: {}".format(evalutation_result[0]))
print("Accuracy: {}".format(evalutation_result[1]))


#Modell benutzen
#test_data = np.array([5.1,3.5,1.4,0.2])
prediction = iris_model.predict(np.array([[5.1,3.5,1.4,0.2]]))
argmax_index = np.argmax(prediction, axis=1)
print("Iris [5.1,3.5,1.4,0.2] gehört zu {}".format(argmax_index))