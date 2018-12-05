# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle

# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'my-activity-data.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i, 1], data[i, 2], data[i, 3]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:, 0:1], reoriented, axis=1)
data = np.append(reoriented_data_with_timestamps, data[:, -1:], axis=1)

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 20

# sampling rate should be about 25 Hz; you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples, 0] - data[0, 0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

# TODO: list the class labels that you collected data for in the order of label_index (defined in collect-labelled-data.py)
class_names = ["sitting", "perfectCurl", "walking", "BadCurl"]  # ...

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []

for i, window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:, 1:-1]
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])

X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)

print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------


# TODO: split data into train and test datasets using 10-fold cross validation

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
k_Fold = 10

kf = KFold(n_splits=k_Fold, shuffle=True)
foldn = 1
for train_index, test_index in kf.split(X, Y):
    print("Fold : " + str(foldn))

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    tree.fit(X_train, Y_train)
    yPrediction = tree.predict(X_test)
    conf = confusion_matrix(Y_test, yPrediction)

    tp = np.diagonal(conf)
    tp_plus_fn = np.sum(conf, axis=0)
    tp_plus_fp = np.sum(conf, axis=1)
    tp_plus_fn_plus_fp = np.subtract(np.add(tp_plus_fn, tp_plus_fp), tp)

    recall = np.divide(tp, tp_plus_fn)
    precision = np.divide(tp, tp_plus_fp)
    accuracy = np.divide(tp, tp_plus_fn_plus_fp)
    print(conf)
    print("TP + FN : " + str(tp_plus_fn))
    print("TP + FP : " + str(tp_plus_fp))
    print("TP + FP + FN: " + str(tp_plus_fn_plus_fp))
    print("TP : " + str(tp))

    print("Accuracy : " + str(accuracy))
    print("Precision : " + str(precision))
    print("Recall : " + str(recall))
    if foldn == 1:
        sumAccuracy = accuracy
        sumRecall = recall
        sumPrecision = precision
    else:
        sumAccuracy += accuracy
        sumRecall += recall
        sumPrecision += precision

    foldn += 1

"""
TODO: iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""

# TODO: calculate and print the average accuracy, precision and recall values over all 10 folds
print("Average Accuracy : " + str(sumAccuracy / k_Fold))
print("Average Precision : " + str(sumPrecision / k_Fold))
print("Average Recall : " + str(sumRecall / k_Fold))

# TODO: train the decision tree classifier on entire dataset
tree.fit(X, Y)

# TODO: Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line
export_graphviz(tree, out_file='tree.dot', feature_names=feature_names)

# TODO: Save the classifier to disk - replace 'tree' with your decision tree and run the below line
with open('classifier.pickle', 'wb') as f:
    pickle.dump(tree, f)