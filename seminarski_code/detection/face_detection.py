import mediapipe as mp
import cv2
import numpy as np

class Face:
    @staticmethod
    def detect_face(frame, model): # passing the single frame from video and model that will proccess it
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB

        frame.flags.writeable = False                  # Image is no longer writeable

        results = model.process(frame)                 # Make prediction

        frame.flags.writeable = True                   # Image is now writeable 

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR

        return frame, results

    @staticmethod
    def get_landmarks(results):
        face_landmarks = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
        return face_landmarks
