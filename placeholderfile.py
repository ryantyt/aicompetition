import cv2
import mediapipe as mp
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time

from kp_ext_func import drawStyledLandmarks, extractKeypoints, mediapipeDetection


# Only if we can't find training data online
DATA_PATH = os.path.join('trainingData')

# Decide on what actions here
actions = np.array([])

# Initialilse mediapipe holistics
mpDrawing = mp.solutions.drawing_utils
mpHands = mp.solutions.hands

# No. sequences and no. frames per video
noSeq = 16
seqLen = 30

labelMap = {label:num for num, label in enumerate(actions)}

# Holistics
mpHolistic = mp.solutions.holistic
mpDrawing = mp.solutions.drawing_utils

sequences, labels = [], []
for action in actions:
    for sequence in range(noSeq):
        window = []
        for frameNum in range(seqLen):
            res = np.load(os.join(DATA_PATH, actions, str(sequence), "{}.py".format(frameNum)))
            window.append(res)

        sequences.append(window)
        labels.append(labelMap[action])

print('Sequences:', sequences)
print('Labels:', labels)

