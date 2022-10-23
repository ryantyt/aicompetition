import cv2
import mediapipe as mp
import numpy as np
import os
import time

from kp_ext_func import drawStyledLandmarks, mediapipeDetection

# Initialilse mediapipe holistics
mpHolistic = mp.solutions.holistic
mpDrawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        width, height = cap.get(3), cap.get(4)

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # results is a list of landmark coordinates
        image, results = mediapipeDetection(frame, holistic)

        drawStyledLandmarks(image, results)

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()