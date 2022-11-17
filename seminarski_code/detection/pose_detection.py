import mediapipe as mp
import cv2
import numpy as np


class Pose:
    @staticmethod
    def detect_pose(frame, model):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        frame.flags.writeable = False                  # Image is no longer writeable
        results = model.process(frame)                 # Make prediction
        frame.flags.writeable = True                   # Image is now writeable 
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return frame, results

    @staticmethod
    def get_landmarks(results):
        pose_landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        return pose_landmarks


        