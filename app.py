# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 19:59:52 2025

@author: jonat
"""

from ImageClassificationClass import ImageRegressor
import gradio as gr

Dog_Model = ImageRegressor()

Dog_Model.LoadModel("Dog_Classifier_Model")

gr.Interface(fn = Dog_Model.ImagePredict, inputs = gr.Image(type = "pil"), outputs = "label").launch()
