# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 22:55:50 2025

@author: jonat
"""

import numpy as np
import h5py
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
import time

def Create_Dataset(Main_File_Name,True_Image_Name, True_File_Name, False_Image_Name, False_File_Name, number_of_pixels, random_state):

    image_files = [f for f in os.listdir(True_File_Name) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    image_data = []
    
    labels = []
    
    dataset_labels = []
    
    print(f"Converting {True_File_Name} files into RGB.")
    for file in tqdm(image_files):
        path = os.path.join(True_File_Name, file)
        img = Image.open(path).resize((number_of_pixels, number_of_pixels)).convert('RGB')  
        img_array = np.array(img)/255  
        image_data.append(img_array)
        labels.append(1)
        dataset_labels.append(True_Image_Name)
        
    print(f"Converting {True_File_Name} files completed.")
    
    image_files = [f for f in os.listdir(False_File_Name) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    time.sleep(0.05)
    
    print(f"Converting {False_File_Name} files into RGB.")
    
    for file in tqdm(image_files):
        path = os.path.join(False_File_Name, file)
        img = Image.open(path).resize((number_of_pixels, number_of_pixels)).convert('RGB')  
        img_array = np.array(img)/255  
        image_data.append(img_array)
        labels.append(0)
        dataset_labels.append(False_Image_Name)
        
    print(f"Converting {False_File_Name} files completed.")
    
    combined = list(zip(image_data, labels, dataset_labels))
    
    combined = shuffle(combined, random_state=random_state)
    
    images_shuffled, class_labels_shuffled, tag_labels_shuffled = zip(*combined)
    
    images_shuffled = np.array(images_shuffled)
    class_labels_shuffled = np.array([class_labels_shuffled])
    tag_labels_shuffled = np.array(tag_labels_shuffled).astype("S")
    
    with h5py.File(Main_File_Name + ".h5", 'w') as h5f:
        string_dt = h5py.string_dtype(encoding='utf-8')
        h5f.create_dataset('set_x', data=images_shuffled)
        h5f.create_dataset('set_y', data=class_labels_shuffled)
        h5f.create_dataset('list_classes', data=tag_labels_shuffled)
        h5f.create_dataset('true_name', data=True_Image_Name, dtype=string_dt )
        h5f.create_dataset('false_name', data=False_Image_Name, dtype=string_dt )

def Show_Image(index, x_array, y_array, classes):
    plt.imshow(x_array[index])
    plt.title(f"It's a {classes[index].decode("utf-8")}, defined as a {str(y_array[0,index])}.", fontsize = 18) 
    
def RGB_Convert(image, num_of_px):
    img_resized = image.resize((num_of_px,num_of_px))
    img_array = np.array(img_resized)/255.0
    return img_array