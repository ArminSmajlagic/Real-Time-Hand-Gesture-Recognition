import mediapipe as mp
import cv2
import numpy as np

class Hands:
    @staticmethod
    def detect_hands(frame, model):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        frame.flags.writeable = False                  # Image is no longer writeable
        results = model.process(frame)                 # Make prediction
        frame.flags.writeable = True                   # Image is now writeable 
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return frame, results

    @staticmethod
    def get_landmarks(results):
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)        
        hands_landmarks = np.array([lh, rh])
        return hands_landmarks