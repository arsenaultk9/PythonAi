# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:40:07 2019

@author: KEARS4
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

data = np.random.random([5, 1])
labels = np.random.random([5, 1])

model = tf.keras.Sequential([
    # input and hidden layers
    layers.Dense(6, activation=tf.nn.relu),
    layers.Dense(6, activation=tf.nn.relu),
    # output layer
    layers.Dense(2, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, epochs=10, batch_size=32)

result = model.predict(data, batch_size=32)
print(result.shape)
