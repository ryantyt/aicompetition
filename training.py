import mediapipe as mp
import cv2
import numpy as np
import os
import sklearn
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

from constants import actions

import preprocessing
sequences = preprocessing.sequences
labelMap = preprocessing.labelMap
labels = preprocessing.labels

X = np.array(sequences)
y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# Softmax gives an array of probabilities that add up to 1
model.add(Dense(actions.shape[0], activation='softmax'))


model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=2000, callbacks = [tb_callback])

model.save('action.h5')