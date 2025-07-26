# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 22:55:50 2025

@author: jonat
"""
from ImageClassificationClass import ImageRegressor

import gradio as gr
from PIL import Image
DogImageRegressor = ImageRegressor()

#DogImageRegressor.Create_Dataset("train_dog_images", "dog", "DogPhotosTraining", "not a dog", "NonDogPhotos", number_of_pixels=16)

x_train, y_train, classification = DogImageRegressor.LoadDataset("train_dog_images.h5")

DogImageRegressor.ShowImage(10)

w,b = DogImageRegressor.TrainModel(learning_rate = 0.05, num_iterations=8000,print_cost=False)

DogImageRegressor.TestModel()

DogImageRegressor.ShowImageTrainingPrediction(1)

DogImageRegressor.ShowImageTestingPrediction(1)

gr.Interface(fn=DogImageRegressor.ImagePredict, inputs=gr.Image(type="pil"), outputs="label").launch()

DogImageRegressor.SaveModel("Dog_Classifier_Model")
