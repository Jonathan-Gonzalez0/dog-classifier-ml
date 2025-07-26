# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 19:59:52 2025

@author: jonathan gonzalez
"""

from ImageClassificationClass import ImageRegressor
import gradio as gr


Dog_Model = ImageRegressor()
Dog_Model.LoadModel("Dog_Classifier_Model")

def predict_fn(image):
    print("Function called!")
    return Dog_Model.ImagePredict(image)

if __name__ == "__main__":
    gr.Interface(
        fn=predict_fn,
        inputs=gr.Image(type="pil"),
        outputs="label"
    ).launch()