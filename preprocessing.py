import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

from constants import DATA_PATH, SEQ_NUM, SEQ_LEN, actions


labelMap = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(SEQ_NUM):
        window = []
        for frameNum in range(SEQ_LEN):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frameNum}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(labelMap[action])

