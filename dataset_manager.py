# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 12:49:38 2025

@author: Jonathan Gonzalez Rodriguez, jonathan.gonzalez57@upr.edu

Image Classifier Dataset Manager

This Python program contains functions used to manipulate data before training.

Last Update: 8/10/2025
"""

def flattened_array(array, samples):
    """ 
    Flattens the input array into shape (pixels, training_samples).

    Parameters
    ----------
    array : np.ndarray
        Array of training samples.
    samples : int
        Total sample size.

    Returns
    -------
    np.ndarray
        Reshaped array.
    """
    return array.reshape(samples,-1).T

def train_test_split(samples, array, ratio):
    """ 
    Divides dataset into training set and testing set.

    Parameters
    ----------
    samples : int
        Total sample size.
    array : np.ndarray
        Array that will be split into training and testing sets.
    ratio : float
        Ratio of training vs testing data.
        For example, 0.8 means 80% training and 20% testing.

    Returns
    -------
    np.ndarray
        Training dataset.
    np.ndarray
        Testing dataset.
    """
    n = int(round(samples*ratio))
    
    training_set = array[:,:n]
    
    testing_set = array[:,n:]
    
    return training_set, testing_set