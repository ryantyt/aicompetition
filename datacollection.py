import cv2
import mediapipe as mp
import numpy as np
import os
import time

from kp_ext_func import drawStyledLandmarks, extractKeypoints, mediapipeDetection

# Only if we can't find training data online

DATA_PATH = os.path.join('trainingData')

# Decide on what actions here
actions = np.array(['a', 'b', 'c'])

# Initialilse mediapipe holistics
mpHolistic = mp.solutions.holistic
mpDrawing = mp.solutions.drawing_utils

# No. sequences and no. frames per video
noSeq = 16
seqLen = 30

# Makes folders if they weren't there already

def main():
    for action in actions:
        for sequence in range(noSeq):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass


    cap = cv2.VideoCapture(0)
    with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(noSeq):
                for frameNum in range(seqLen):

                    width, height = cap.get(3), cap.get(4)

                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)

                    # results is a list of landmark coordinates
                    image, results = mediapipeDetection(frame, holistic)

                    drawStyledLandmarks(image, results)

                    if frameNum == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (width/2, height/2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Action: {action}, Video Number: {sequence}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, f'Action: {action}, Video Number: {sequence}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)

                    keypoints = extractKeypoints(results)
                    npyPath = os.path.join(DATA_PATH, action, str(sequence), str(frameNum))
                    np.save(npyPath, keypoints)

                    cv2.imshow('Hand Tracking', image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "main":
    main()