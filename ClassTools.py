# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 17:06:20 2025

@author: jonat
"""
import numpy as np
from tqdm import tqdm

def sigmoid(z):
    return 1/(1+np.exp(-z))

def propagate(w,b,x,y,m):
    a = sigmoid(np.dot(w.T,x)+b)
    cost = (-1/m)*np.sum(y*np.log(a) + (1-y)*np.log(1-a))
    dw = (1/m)*np.dot(x,(a-y).T)
    db = (1/m)*np.sum(a-y)
    
    cost = np.array(np.squeeze(cost))
    
    grads = {"dw": dw, "db":db}
    
    return grads, cost

def optimize(w,b,x,y,m,number_of_iteratons,learning_rate,print_cost,):
    
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

