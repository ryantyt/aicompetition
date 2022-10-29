import cv2
import mediapipe as mp
import numpy as np
import os
import errno

from kp_ext_func import drawStyledLandmarks, extractKeypoints, mediapipeDetection
from constants import DATA_PATH, SEQ_NUM, SEQ_LEN, actions

# Initialilse mediapipe holistics
mpHolistic = mp.solutions.holistic
mpDrawing = mp.solutions.drawing_utils

def main():

    # Makes folders if they weren'T there already
    for action in actions:
        for sequence in range(SEQ_NUM):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise


    cap = cv2.VideoCapture(0)
    width, height = cap.get(3), cap.get(4)
    x, y = int(width)/2, int(height)/2
    print(type(width))

    with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(SEQ_NUM):
                for frameNum in range(SEQ_LEN):

                    _, frame = cap.read()
                    frame = cv2.flip(frame, 1)

                    # results is a list of landmark coordinates
                    image, results = mediapipeDetection(frame, holistic)

                    drawStyledLandmarks(image, results)

                    cv2.putText(image, f'Action: {action}, Video Number: {sequence + 1}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)

                    if frameNum == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (640, 480), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.waitKey(5000)

                    keypoints = extractKeypoints(results)
                    npyPath = os.path.join(DATA_PATH, action, str(sequence), str(frameNum))
                    np.save(npyPath, keypoints)

                    cv2.imshow('Hand Tracking', image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()