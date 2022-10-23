from tkinter import N
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import sklearn
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard


import datacollection

# find a way to share the variables first before continuing 

labelMap = {label:num for num, label in enumerate(datacollection.actions)}

print(labelMap)
