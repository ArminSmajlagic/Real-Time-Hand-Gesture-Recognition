import os
import time
import numpy as np

start_folder = 30

data_path = os.path.join('data') # path where i hold my videos
actions = np.array(['like','ok','hello']) # gestures that i detect
no_sequences = 30 # number of videos that
sequence_length = 30 # length of the video in frames

class FileManager:
    @staticmethod
    def create_folders(): # static method responsible for creating folders where i wil hold my videos - data for training
        for action in actions:
            for sequence in range(1,no_sequences+1):
                try: 
                    os.makedirs(os.path.join(data_path, action, str(sequence))) # this wil create file data/action/n ; where n is the sequence
                except:
                    pass

    @staticmethod
    def add_video_to_file(keypoints, action, sequence, frame_no):
        npy_path = os.path.join(data_path, action, sequence, frame_no)
        np.save(npy_path, keypoints)
