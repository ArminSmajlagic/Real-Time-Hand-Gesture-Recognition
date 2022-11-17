import os
import time
import numpy as np

start_folder = 30

data_path = os.path.join('data') # path where i will hold my videos
actions = np.array(['like','ok','hello']) # gestures that i will detect
no_sequences = 30 # number of videos
sequence_length = 30 # length of the video in frames

class FileManager:
    @staticmethod
    def create_folders():
        print('this happened')
        for action in actions: 
            for sequence in range(1,no_sequences+1):
                try: 
                    os.makedirs(os.path.join(data_path, action, str(sequence)))
                except:
                    pass

    @staticmethod
    def add_video_to_file(keypoints, action, sequence, frame_no):
        npy_path = os.path.join(data_path, action, sequence, frame_no)
        np.save(npy_path, keypoints)
