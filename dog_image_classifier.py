# -*- coding: utf-8 -*-
"""
Created on Mon August 6 22:55:50 2025

@author: Jonathan Gonzalez Rodriguez, jonathan.gonzalez57@upr.edu

Image Classifier Program

This Python program was developed to demonstrate the functionality of a 
custom built image classifier class. The class provides core methods for 
dataset preparation, model training, evaluation, and prediction.

The program not only defines the class but also includes an example workflow 
that uses the class as intended from loading image data and training a model 
to evaluating its performance and generating predictions.

Last Update: 8/22/2025
"""
from image_classification_class import ImageRegressor
import matplotlib.pyplot as plt

plt.close("all")

dog_image_regressor = ImageRegressor()

dog_image_regressor.create_dataset("train_dog_images", "Dog", "dog_photos", "Not a Dog", "non_dog_photos", number_of_pixels=16)

x_train, y_train, classification = dog_image_regressor.load_dataset("train_dog_images.h5")

dog_image_regressor.show_image(10)

w, b = dog_image_regressor.train_model(learning_rate = 0.05, epochs=2000,print_cost=False)

dog_image_regressor.test_model()

dog_image_regressor.show_image_testing_prediction(1) # Show image prediction from index 1 to index 1611

dog_image_regressor.confusion_matrix()

dog_image_regressor.save_model("dog_classifier_model")
