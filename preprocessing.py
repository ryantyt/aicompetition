import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

# Sharing variables
import datacollection
actions = datacollection.actions
noSeq = datacollection.noSeq
seqLen = datacollection.seqLen
DATA_PATH = datacollection.DATA_PATH

# # find a way to share the variables first before continuing 

labelMap = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(noSeq):
        window = []
        for frameNum in range(seqLen):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frameNum}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(labelMap[action])

