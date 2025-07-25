# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 12:49:38 2025

@author: jonat
"""

import numpy as np
import h5py

def load_dataset(file_name):
    dataset = h5py.File(file_name, 'r')
    x_array = np.array(dataset["set_x"][:])
    y_array = np.array(dataset["set_y"][:])
    classification = np.array(dataset["list_classes"][:])
    true_name = dataset["true_name"][()].decode()
    false_name = dataset["false_name"][()].decode()
    
    return x_array, y_array, classification, true_name, false_name

def flattened_array(array, samples):
    return array.reshape(samples,-1).T

def initialize_parameters(dimensions):
    return np.zeros((dimensions,1)), 0

def train_test_split(samples, array, ratio):
    n = int(round(samples*ratio))
    return array[:,:n], array[:,n:]