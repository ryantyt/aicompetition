from msilib import sequence
from cv2 import KeyPoint
from matplotlib.pyplot import text
import mediapipe as mp
import cv2
import numpy as np
import os
import sklearn
import tensorflow as tf
import PySimpleGUI as sg

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Functions and constnats
from kp_ext_func import drawStyledLandmarks, extractKeypoints, mediapipeDetection
from constants import actions, THRESHOLD

mpHolistic = mp.solutions.holistic # Holistic model
mpDrawing = mp.solutions.drawing_utils # Drawing utilities

# Loading LSTM Neural Network
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# Softmax gives an array of probabilities that add up to 1
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights('action.h5')

text_col = sg.Column([
    [
        sg.Text('Sentence Starts Here', key='WORDS', expand_x=True, justification='r', font='Calibri 20', border_width=1)
        ]
])
layout = [
    [sg.Image(key='IMAGE'), text_col]
]
# Storing Keypoints
sequence = []

window = sg.Window('ASL Detection', layout, return_keyboard_events=True)

# window with camera, text displayed
# other tab that can display code
# live text updating

cap = cv2.VideoCapture(0)
with mpHolistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5) as holistic:
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Escape:889192475': break
        if event != '__TIMEOUT__': print(event)

        # Detection Logic
        _, frame = cv2.read()
        frame = cv2.flip(frame, 1)

        # Self explanatory
        img, results = mediapipeDetection(frame, holistic)
        keypoints = extractKeypoints(results)
        drawStyledLandmarks(img, results)

        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Displays the sign if detection is above threshold
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            if res[np.argmax(res)] >= THRESHOLD:
                window['WORDS'].update(values['WORDS'] + ' ' + actions[np.argmax(res)])

        # Updates the image inside the pysimplegui window
        disp = cv2.imencode('.png', img)[1].tobytes()
        window['IMAGE'].update(data=disp)

window.close()
cv2.destroyAllWindows()