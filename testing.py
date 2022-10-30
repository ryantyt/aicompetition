import mediapipe as mp
import cv2
import numpy as np
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

from constants import THRESHOLD

from kp_ext_func import drawLandmarks, extractKeypoints, mediapipeDetection

import datacollection
actions = datacollection.actions

mpHolistic = mp.solutions.holistic # Holistic model
mpDrawing = mp.solutions.drawing_utils # Drawing utilities

sequence, sentence = [], []

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# Softmax gives an array of probabilities that add up to 1
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights('action.h5')

def main(sequence, sentence):
    # update('start typing')

    cap = cv2.VideoCapture(0)
    with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image, results = mediapipeDetection(frame, holistic)

            keypoints = extractKeypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

                if res[np.argmax(res)] > THRESHOLD:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        print(np.argmax(res))
                        sentence.append(actions[np.argmax(res)])
                
                if len(sentence) > 10:
                    sentence = sentence[-10:]
            drawLandmarks(image, results)
            cv2.rectangle(image, (0, 0), (1280, 40), (0,0,0), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sequence, sentence)