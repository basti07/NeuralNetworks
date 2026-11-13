#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 21:27:47 2022

@author: basti07
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Layer, Input, MaxPooling2D, Convolution2D, Flatten, Dense, Concatenate
import os
import numpy as np
from PIL import Image
import random
# import tensorflowjs as tfjs


# lfw-Bilder bewegen (nur ein Mal)
# for directories in os.listdir("siamese_dataset/negatives/lfw"):
#     for files in os.listdir(os.path.join("siamese_dataset/negatives/lfw", directories)):
#         old_path = os.path.join("siamese_dataset/negatives/lfw", directories, files)
#         new_path = os.path.join("siamese_dataset/negatives", files)
#         os.replace(old_path, new_path)


# Daten vorbereiten
def data_generator(batch_size=64):
    while True:
        a = []
        p = []
        n = []
        for i in range(batch_size):
            a_path = "siamese_dataset/anchor"
            p_path = "siamese_dataset/positives"
            n_path = "siamese_dataset/negatives"
            
            a_files = os.listdir(a_path)
            a_pic_path = random.choice(a_files)
            p_files = os.listdir(p_path)
            p_pic_path = random.choice(p_files)
            n_files = os.listdir(n_path)
            n_pic_path = random.choice(n_files)
            
            a_pic = Image.open(a_path + "/" + a_pic_path).resize((28,28))
            a_pic_data = np.array(a_pic)
            p_pic = Image.open(p_path + "/" + p_pic_path).resize((28,28))
            p_pic_data = np.array(p_pic)
            n_pic = Image.open(n_path + "/" + n_pic_path).resize((28,28))
            n_pic_data = np.array(n_pic)
            
            # a_pic_data.reshape(200, 200, 3)
            # p_pic_data.reshape(200, 200, 3)
            # n_pic_data.reshape(200, 200, 3)
            
            a.append(a_pic_data)
            p.append(p_pic_data)
            n.append(n_pic_data)
            
        yield ([np.array(a), np.array(p), np.array(n)], np.zeros((batch_size, 1)).astype("float32"))
        
# Loss Funktion
class TripletLoss(Loss):
 
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
 
    def call(self, y_true, y_pred):
        anchor = y_pred[:, 0:100]
        pos    = y_pred[:, 100:200]
        neg    = y_pred[:, 200:300]
        pos_dist   = tf.reduce_sum(tf.square(tf.subtract(anchor, pos)), axis=-1)    
        neg_dist   = tf.reduce_sum(tf.square(tf.subtract(anchor, neg)), axis=-1)    
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.margin)    
        loss       = tf.reduce_sum(tf.maximum(basic_loss, 0.0))               
        return loss
    def get_config(self):
        config = super(TripletLoss, self).get_config()
        return {**config, 'margin': self.margin}
    


# Modell erstellen
# Schwester-Block
def make_sister_model():
    inp = Input(shape=(28,28,3), name="input_layer")
    # conv1 = Convolution2D(64, (10,10), activation="relu")(inp)
    # pool1 = MaxPooling2D(64, (2,2), padding="same")(conv1)
    # conv2 = Convolution2D(128, (7,7), activation="relu")(pool1)
    # pool2 = MaxPooling2D(64, (2,2), padding="same")(conv2)
    # conv3 = Convolution2D(128, (4,4), activation="relu")(pool2)
    # pool3 = MaxPooling2D(64, (2,2), padding="same")(conv3)
    # conv4 = Convolution2D(256, (4,4), activation="relu")(pool3)
    # flt1 = Flatten()(conv4)
    # den1 = Dense(4096, activation="relu")(flt1)
    conv1 = Convolution2D(32, 3, activation="relu")(inp)
    conv2 = Convolution2D(32, 3, activation="relu")(conv1)
    pool1 = MaxPooling2D(2)(conv2)
    conv3 = Convolution2D(64, 3, activation="relu")(pool1)
    conv4 = Convolution2D(64, 3, activation="relu")(conv3)
    pool2 = MaxPooling2D(2)(conv4)
    conv5 = Convolution2D(128, 3, activation="relu")(pool2)
    flt1 = Flatten()(conv5)
    den1 = Dense(100, activation="relu")(flt1)
    
    
    return Model(inputs=[inp], outputs=[den1], name="sister_model")

sister_model = make_sister_model()



# Die Schwester-Modelle erstellen
input_inp = Input(shape=(28,28,3))
input_pos = Input(shape=(28,28,3))
input_neg = Input(shape=(28,28,3))

out = Concatenate()([sister_model(input_inp), sister_model(input_pos), sister_model(input_neg)])
model = Model([input_inp, input_pos, input_neg], out)
model.summary()

model.compile(loss=TripletLoss(0.5), optimizer="adam")
model.fit(data_generator(), steps_per_epoch=150, epochs=3, verbose=1)

# model.save_weights("triplet_loss")
model.save("export/saved_model.hdf5")
# tfjs.converters.save_keras_model(model, "export")




