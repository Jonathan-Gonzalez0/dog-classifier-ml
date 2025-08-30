# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 17:06:20 2025

@author: Jonathan Gonzalez Rodriguez, jonathan.gonzalez57@upr.edu

Image Classifier Class Tools

This Python program contains tools used by the class to train the 
model.

Last Update: 8/10/2025
"""
import numpy as np
from tqdm import tqdm

def sigmoid(z):
    """ 
    Applies the sigmoid activation function: 1 / (1 + exp(-x))

    Parameters
    ----------
    z : np.ndarray
        Expression to be used as the input (z) for the sigmoid activation 
        function.

    Returns
    -------
    sigmoid : np.ndarray
        NumPy array containing the evaluation of the sigmoid function for the 
        given inputs.
    """
    
    sigmoid = 1/(1+np.exp(-z))
    
    return sigmoid

def propagate(w,b,x,y,m):
    """ 
    Calculates updated cost, weights, and bias for the model.

    Parameters
    ----------
    w : np.ndarray
        Current weights of the model.
    b : float
        Current bias of the model.
    x : np.ndarray
        Training input data (features) used to fit the model.
    y : np.ndarray
        True target values (labels).
    m : int
        Sample size (number of training examples). 

    Returns
    -------
    grads : dict
        Dictionary containing the weights and bias.
    """
    a = sigmoid(np.dot(w.T,x)+b)
    cost = (-1/m)*np.sum(y*np.log(a) + (1-y)*np.log(1-a))
    dw = (1/m)*np.dot(x,(a-y).T)
    db = (1/m)*np.sum(a-y)
    
    cost = np.array(np.squeeze(cost))
    
    grads = {"dw": dw, "db":db}
    
    return grads, cost

def optimize(w,b,x,y,m,number_of_iteratons,learning_rate,print_cost,):
    """ 
    Calculates updated cost, weights, and bias for the model.

    Parameters
    ----------
    w : np.ndarray
        Current weights of the model.
    b : float
        Current bias of the model.
    x : np.ndarray
        Training input data (features) used to fit the model.
    y : np.ndarray
        True target values (labels).
    m : int
        Sample size (number of training examples). 

    Returns
    -------
    grads : dict
        Dictionary containing the weights and bias.
    """
    costs = []

    for i in tqdm(range(number_of_iteratons)):
        grads, cost = propagate(w, b, x, y, m)
        costs.append(cost)
        dw = grads["dw"]
        db = grads["db"]
        
        w -= learning_rate*dw
        b -= learning_rate*db
        
        
        if i%100 == 0 and print_cost:
            print("Cost after iteration %i: %f" %(i,cost))
    params = {"w":w, "b":b}
    
    grads = {"dw": dw, "db":db}
    
    return params, grads, costs


def initialize_parameters(weight_size):
    """ 
    Initialies parameters as zeros.

    Parameters
    ----------
    weight_size : int
        Numer of weights used in the model.

    Returns
    -------
    w : np.ndarray
        Weights initialized as zero. 
    b : float
        Bias initialized as zero.
    """
    
    w = np.zeros((weight_size,1))
    
    b = 0
    
    return w, b

