# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 19:59:52 2025

@author: Jonathan gonzalez
"""

import gradio as gr

def hello(image):
    return "Hello!"

gr.Interface(fn=hello, inputs=gr.Image(type="pil"), outputs="label").launch(debug=True)