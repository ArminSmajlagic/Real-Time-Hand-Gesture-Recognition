from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.callbacks import TensorBoard


import os
import numpy as np

import helper.data_files_manager as dfm

log_dir = os.path.join('training-logs')
tb_callback = TensorBoard(log_dir=log_dir)

class Trainer:
    model = Sequential()

    @staticmethod
    def preprocces_data(sequences,labels):

        X = np.array(sequences) # sequence of data - the previously caputred 30x30x3 data

        y = to_categorical(labels).astype(int) # generating y - labels - output

        x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.05) # spliting the outputs and inputs into test and train data

        return x_train, x_test, y_train, y_test

    @staticmethod
    def create_labels(actions): 

        label_map = {label:num for num, label in enumerate(actions)} # creating dictionary map of actions

        sequences, labels = [], []

        # looping through actions that i want to detect and data i previously captured and storing the sequence data and labels
        for action in actions:
            for sequence in range(1,30):
                window = []
                for frame_num in range(0,29):
                    res = np.load(os.path.join(dfm.data_path, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])

        return sequences, labels

    @staticmethod
    def train(actions):

        sequences, labels = Trainer.create_labels(actions) # Creating labels from actions

        x_train, x_test, y_train, y_test = Trainer.preprocces_data(sequences, labels)  # Preparing x and y, test and train data
        
        model = Sequential() # Defining our feedforward neural network

        # adding layeres to our netowrk - total 6 layers - 1 input ; 4 hidden ; 1 output

        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) # Long-Short Memory layer - 64 Neurons - Activation ReLU
        model.add(LSTM(128, return_sequences=True, activation='relu')) # Long-Short Memory layer - 128 Neurons - Activation ReLU
        model.add(LSTM(64, return_sequences=False, activation='relu')) # Long-Short Memory layer - 64 Neurons - Activation ReLU
        model.add(Dense(64, activation='relu')) # Dense layer - 64 Neurons - Activation ReLU
        model.add(Dense(32, activation='relu')) # Dense layer - 32 Neurons - Activation ReLU
        model.add(Dense(3, activation='softmax')) # Dense layer - 3 Neurons - Activation sofmax

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy']) # compiling the NN

        model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback]) # fiting NN to my data

        model.save('action2.h5') # saving the NN for later use





