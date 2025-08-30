# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 22:55:50 2025

@author: Jonathan Gonzalez Rodriguez, jonathan.gonzalez57@upr.edu

Image Classifier Class Image Manager

Provides image manipulation utilities used by the classifier class to 
preprocess data and assist in training the model.

Last Update: 8/10/2025
"""

import numpy as np
import h5py
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
import time

def create_rgb_dataset(main_file_name,true_image_name, true_file_name, false_image_name, false_file_name, number_of_pixels, random_state):
    """ 
    Accesses the true and false image dataset folders, converts images to RGB, 
    normalizes them, and creates a randomized dataset.

    Parameters
    ----------
    main_file_name : str
        Variable used as a name to store the RGB converted training set in a 
        .h5 file.
    true_image_name : str
        Name given to true images.
    true_file_name : str
        Folder containing the true images.
    false_image_name : str
        Name given to false images.
    false_file_name : str
        Folder containing the false images.
    number_of_pixels : int
        User defined number of pixels per image for RGB conversion. 
    random_state : int
        Seed value used to ensure reproducibility of random operations. 
        
    Returns
    -------
    None.
    """
    image_files = [f for f in os.listdir(true_file_name) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    image_data = []
    
    labels = []
    
    dataset_labels = []
    
    print(f"Converting {true_file_name} files into RGB.")
    for file in tqdm(image_files):
        path = os.path.join(true_file_name, file)
        img = Image.open(path).resize((number_of_pixels, number_of_pixels)).convert('RGB')  
        img_array = np.array(img)/255  
        image_data.append(img_array)
        labels.append(1)
        dataset_labels.append(true_image_name)
        
    print(f"Converting {true_file_name} files completed.")
    
    image_files = [f for f in os.listdir(false_file_name) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    time.sleep(0.05)
    
    print(f"Converting {false_file_name} files into RGB.")
    
    for file in tqdm(image_files):
        path = os.path.join(false_file_name, file)
        img = Image.open(path).resize((number_of_pixels, number_of_pixels)).convert('RGB')  
        img_array = np.array(img)/255  
        image_data.append(img_array)
        labels.append(0)
        dataset_labels.append(false_image_name)
        
    print(f"Converting {false_file_name} files completed.")
    
    combined = list(zip(image_data, labels, dataset_labels))
    
    combined = shuffle(combined, random_state=random_state)
    
    images_shuffled, class_labels_shuffled, tag_labels_shuffled = zip(*combined)
    
    images_shuffled = np.array(images_shuffled)
    class_labels_shuffled = np.array([class_labels_shuffled])
    tag_labels_shuffled = np.array(tag_labels_shuffled).astype("S")
    
    with h5py.File(main_file_name + ".h5", 'w') as h5f:
        string_dt = h5py.string_dtype(encoding='utf-8')
        h5f.create_dataset('set_x', data=images_shuffled)
        h5f.create_dataset('set_y', data=class_labels_shuffled)
        h5f.create_dataset('list_classes_names', data=tag_labels_shuffled)
        h5f.create_dataset('true_name', data=true_image_name, dtype=string_dt )
        h5f.create_dataset('false_name', data=false_image_name, dtype=string_dt )

def show_image(index, x_array, y_array, classes_name):
    """ 
    Displays the image from the dataset array at the specified index.
    
    Parameters
    ----------
    index : int
        Index of the dataset array.
    x_array : np.ndarray
        Array containing the images.
    y_array : np.ndarray
        Array containing the classification labels.
    classes_name : np.ndarray
        Array containing labels of the images.

    Returns
    -------
    None.
    """
    plt.figure()
    plt.imshow(x_array[index])
    plt.title(f"It's a {classes_name[index].decode('utf-8')}, defined as a {str(y_array[0,index])}.", fontsize = 18) 
    plt.show()
    
def rgb_convert(image, num_of_px):
    """ 
    Converts the image to RGB normalized format using the given pixel 
    dimensions.
    
    Parameters
    ----------
    image : PIL.Image.Image
        Input image to be converted to RGB.
    num_px : int
        Pixel dimension.

    Returns
    -------
    img_array : np.ndarray
        Array containing the RGB image normalized to the range [0, 1].
    """
    img_resized = image.resize((num_of_px,num_of_px))
    img_array = np.array(img_resized)/255.0
    return img_array