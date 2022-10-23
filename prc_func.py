import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def getLabel(index, hand, results, width, height):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:

            label = classification.classification[0].index
            score = classification.classification[0].score

            # Text is labels and score
            text = '{} {}'.format(label, round(score, 2))

            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)), 
                [width, height]).astype(int))

            output = text, coords

    return output

def drawFingerAngles(image, results, jointList, width, height):

    for hand in results.multi_hand_landmarks:
        for joint in jointList:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians*180.0/np.pi)

            if angle > 180.0:
                angle = 360 - angle

            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [width, height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return image