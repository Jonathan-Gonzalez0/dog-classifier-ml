# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 22:16:23 2025

@author: jonat
"""

from ImageClassificationClass import ImageRegressor


Dog_Model = ImageRegressor()
Dog_Model.LoadModel("Dog_Classifier_Model")
Dog_Model.TestModel()
Dog_Model.ConfusionMatrix()
Dog_Model.ShowImageTestingPrediction(5)

