# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 19:59:52 2025

@author: jonathan gonzalez
"""

from ImageClassificationClass import ImageRegressor
import gradio as gr

def predict_fn(image):
    print("Function called!")
    return "Test output"

gr.Interface(fn=predict_fn, inputs=gr.Image(type="pil"), outputs="label").launch(debug=True)