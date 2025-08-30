# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 19:59:52 2025

@author: jonathan gonzalez
"""

from image_classification_class import ImageRegressor
import gradio as gr


dog_model = ImageRegressor()
dog_model.load_model("dog_model")

def predict_fn(image):
    return dog_model.image_predict(image)

gr.Interface(
    fn=predict_fn,
    inputs=gr.Image(type="pil"),
    outputs="label"
).launch()