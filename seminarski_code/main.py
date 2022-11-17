import cv2
import numpy as np
import mediapipe as mp
from scipy import stats

import detection.face_detection as face_detector
import detection.hands_detection as hands_detector
import detection.pose_detection as pose_detector
import helper.landmark_drawer as drawer
import helper.data_files_manager as dfm
import training.trainer as Trainer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.callbacks import TensorBoard
mp_holistic = mp.solutions.holistic # Holistic model

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

def run_camera():
    cap = cv2.VideoCapture(0)
    return cap

def show_capture_image(image):
    cv2.imshow('OpenCV Feed', image)

def extract_keypoints(results):
    face_landmarks = face_detector.Face.get_landmarks(results)
    pose_landmarks = pose_detector.Pose.get_landmarks(results)
    left_hand_landmarks = hands_detector.Hands.get_landmarks(results)[0]
    right_hand_landmarks = hands_detector.Hands.get_landmarks(results)[1]
    return np.concatenate([pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks])

def Main():
    cap = run_camera()
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections - here i detect face,pose and hands using detectors
            frame, results = face_detector.Face.detect_face(frame, holistic)
            frame, results = hands_detector.Hands.detect_hands(frame, holistic)
            frame, results = pose_detector.Pose.detect_pose(frame, holistic)
            
            # Draw landmarks - here i draw landmarks using landmark_drawer
            drawer.LandmarkDrawer.draw_face_landmarks(frame, results)
            drawer.LandmarkDrawer.draw_hands_landmarks(frame, results)
            drawer.LandmarkDrawer.draw_pose_landmarks(frame, results)

            # Extracting ladnmarks
            landmarks = extract_keypoints(results)

            show_capture_image(frame)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

def capture_sequnce():
    cap = run_camera()
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in dfm.actions:
            for sequence in range(1,dfm.no_sequences+1):
                for frame_num in range(dfm.sequence_length):

                    ret, frame = cap.read()

                    # Make detections - here i detect face,pose and hands using detectors
                    #frame, results = face_detector.Face.detect_face(frame, holistic)
                    frame, results = hands_detector.Hands.detect_hands(frame, holistic)
                    #frame, results = pose_detector.Pose.detect_pose(frame, holistic)
                    
                    # Draw landmarks - here i draw landmarks using landmark_drawer
                    # drawer.LandmarkDrawer.draw_face_landmarks(frame, results)
                    drawer.LandmarkDrawer.draw_hands_landmarks(frame, results)
                    # drawer.LandmarkDrawer.draw_pose_landmarks(frame, results)
           
                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(frame, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    show_capture_image(frame)
                    
                    keypoints = extract_keypoints(results)
                    dfm.FileManager.add_video_to_file(keypoints, action, str(sequence), str(frame_num))

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
    cap.release()
    cv2.destroyAllWindows()



def final():
        # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7

    model = Sequential()

    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1900)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)   
     # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, image = cap.read()

            # Make detections
            image, results = face_detector.Face.detect_face(image, holistic)
            image, results = hands_detector.Hands.detect_hands(image, holistic)
            image, results = pose_detector.Pose.detect_pose(image, holistic)        
            
            # Draw landmarks
            #drawer.LandmarkDrawer.draw_hands_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.insert(0, keypoints)
            sequence = sequence[:30]


            model.load_weights('action.h5')

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                #print(dfm.actions[np.argmax(res)])
                
                
                # #3. Viz logic
                # if res[np.argmax(res)] > threshold: 
                #     if len(sentence) > 0: 
                #         if dfm.actions[np.argmax(res)] != sentence[-1]:
                #             sentence.append(dfm.actions[np.argmax(res)])
                #     else:
                #         sentence.append(dfm.actions[np.argmax(res)])

                # if len(sentence) > 5: 
                #     sentence = sentence[-5:]

            # # Viz probabilities
                image = prob_viz(res, dfm.actions, image, colors)
                
            # cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            # cv2.putText(image, ' '.join(sentence), (3,30), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()    


def refined():

    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    model = Sequential()

    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(64, return_sequences=False, activation='relu', input_shape=(30, 1662)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(dfm.actions.shape[0], activation='softmax')) # 3 output nodes for 3 actions

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            success, image = cap.read()

            # Make detections
            image, results = hands_detector.Hands.detect_hands(image, holistic)
            
            # # Draw landmarks
            # draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(dfm.actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                
                #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if dfm.actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(dfm.actions[np.argmax(res)])
                        else:
                            sentence.append(dfm.actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, dfm.actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()