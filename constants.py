import os
import numpy as np


# Only if we can'T find training data online
DATA_PATH = os.path.join('trainingData')

# No. sequences and no. frames per video
SEQ_NUM = 16
SEQ_LEN = 30

THRESHOLD = 0.8

# Decide on what actions here
actions = np.array([])

# Constants
T = 2
T1 = 3
R = 2
