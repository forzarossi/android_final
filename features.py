# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
import scipy.signal as sig
import scipy.stats as stat
from scipy.stats import entropy



def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window.
    """
    return np.mean(window, axis=0)


# TODO: define functions to compute more features

# standard deviation
def _compute_std_dev(window):
    return np.std(window, axis=0)


# peaks or zero crossings
def _compute_peaks(window):
    peaks = []
    peaks.append(len(sig.find_peaks(window[:, 0], prominence=1)))
    peaks.append(len(sig.find_peaks(window[:, 1], prominence=1)))
    peaks.append(len(sig.find_peaks(window[:, 2], prominence=1)))
    return peaks

def _compute_frequency(window):
    freq = []
    peaks = _compute_peaks(window)
    freq.append(.8/peaks[0])
    freq.append(.8 / peaks[1])
    freq.append(.8 / peaks[2])
    return freq

# max
def _compute_max(window):
    return np.max(window,axis=0)


# min
def _compute_min(window):
    return np.min(window,axis=0)


# variance
def _compute_variance(window):
    return np.var(window, axis=0)


# real-valued discrete fourier transform
def _compute_fourierT(window):
    a = np.fft.rfft(window, axis=0).astype(float)
    return a.ravel()


# entropy
def _compute_entropy(window):
    # data = window.value_counts() / len(window)
    # return stat.entropy(data)
    value, counts = np.unique(window, return_counts=True)
    return entropy(counts, base=None)


# velocity
def _compute_velocity(window):
    velocity = []
    velocity_x = 0
    velocity_y = 0
    velocity_z = 0

    for a in window[:,0]:
        velocity_x += a * .04
    for a in window[:, 1]:
        velocity_y += a * .04
    for a in window[:, 2]:
        velocity_z += a * .04

    velocity.append(velocity_x)
    velocity.append(velocity_y)
    velocity.append(velocity_z)

    return velocity


# distance
def _compute_distance(window):
    distance = []
    distance_x = 0
    distance_y = 0
    distance_z = 0
    Vnot = _compute_velocity(window[::2])

    for a in window[:,0]:
        distance_x += .5 * a * (.04* .04) + Vnot[0] * .04
    for a in window[:, 1]:
        distance_y += .5 * a * (.04 * .04) + Vnot[1] * .04
    for a in window[:, 2]:
        distance_z += .5 * a * (.04 * .04) + Vnot[2] * .04
    distance.append(distance_x)
    distance.append(distance_y)
    distance.append(distance_z)
    return distance



def extract_features(window):
    """
    Here is where you will extract your features from the data over
    the given window. We have given you an example of computing
    the mean and appending it to the feature vector.

    """

    x = []
    feature_names = []
    #print (window)
    x.append(_compute_mean_features(window))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")

    x.append(_compute_std_dev(window))
    feature_names.append("x_std_dev")
    feature_names.append("y_std_dev")
    feature_names.append("z_std_dev")

    x.append(_compute_peaks(window))
    feature_names.append("x_peaks")
    feature_names.append("y_peaks")
    feature_names.append("z_peaks")

    x.append(_compute_frequency(window))
    feature_names.append("x_frequency")
    feature_names.append("y_frequency")
    feature_names.append("z_frequency")

    x.append(_compute_min(window))
    feature_names.append("x_min")
    feature_names.append("y_min")
    feature_names.append("z_min")

    x.append(_compute_max(window))
    feature_names.append("x_max")
    feature_names.append("y_max")
    feature_names.append("z_max")
    
    x.append(_compute_variance(window))
    feature_names.append("x_variance")
    feature_names.append("y_variance")
    feature_names.append("z_variance")

    # print("Starting here")
    # print(_compute_fourierT(window))
    # x.append(_compute_fourierT(window))
    # feature_names.append("x_fourierT")
    # feature_names.append("y_fourierT")
    # feature_names.append("z_fourierT")

    # x.append(_compute_entropy(window))
    # feature_names.append("entropy")

    x.append(_compute_velocity(window))
    feature_names.append("x_velocity")
    feature_names.append("y_velocity")
    feature_names.append("z_velocity")

    x.append(_compute_distance(window))
    feature_names.append("x_distance")
    feature_names.append("y_distance")
    feature_names.append("z_distance")

    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names

    feature_vector = np.concatenate(x, axis=0)  # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector