# -*- coding: utf-8 -*-
"""
This Python script receives incoming unlabelled accelerometer data through 
the server and uses your trained classifier to predict its class label.

"""

import socket
import sys
import json
import threading
import numpy as np
import pickle
from features import extract_features # make sure features.py is in the same directory
from util import reorient, reset_vars

# TODO: list the class labels that you collected data for in the order of label_index (defined in collect-labelled-data.py)
class_names = ["sitting", "perfectCurl", "walking", "badcurl-elbowup", "half_curl", "back_elbow", "high_half", "elbow_out", "elbow_up", "elbow_back", "resting", "perfectCurl"] #...
# label numbers  0              1           2              3                4           5               6           7           8           9            10            11
activity = "none"
count = 0

# Loading the classifier that you saved to disk previously
with open('classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)
    
if classifier == None:
    print("Classifier is null; make sure you have trained it!")
    sys.exit()
    
def onActivityDetected(activity):
    """
    Notifies the user of the current activity
    """
    return activity

def predict(window):
    """
    Given a window of accelerometer data, predict the activity label. 
    """
    global activity
    # TODO: extract features over the window of data
    feature_names, feature_vector = extract_features(window)
    
    # TODO: use classifier.predict(feature_vector) to predict the class label.
    # Make sure your feature vector is passed in the expected format
    # print(np.reshape(feature_vector, (1, -1)))
    label = classifier.predict(np.reshape(feature_vector, (1, -1)))

    # print(label)
    
    # TODO: get the name of your predicted activity from 'class_names' using the returned label.
    # pass the activity name to onActivityDetected()
    activity = onActivityDetected(class_names[int(label[0])])
    return
