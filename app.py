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

from kp_ext_func import extractKeypoints, mediapipeDetection

import datacollection
actions = datacollection.actions

mpHolistic = mp.solutions.holistic # Holistic model
mpDrawing = mp.solutions.drawing_utils # Drawing utilities

sequence, sentence = [], []

threshold = 0.4

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# Softmax gives an array of probabilities that add up to 1
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights('action.h5')

