# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 14:14:40 2025

@author: jonat
"""
import ImageManager as im
import DatasetManager as dm
import ClassTools as ct
import numpy as np
import h5py
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ImageRegressor:
    
    def __init__(self):
        self.__x_array = []
        self.__training_x_array = []
        self.__testing_x_array = []
        self.__y_array = []
        self.__training_y_array = []
        self.__testing_y_array = []
        self.__testing_predictions = []
        self.__training_predictions = []
        self.__classes = []
        self.__training_classes = []
        self.__testing_classes = []
        self.__training_samples = 0
        self.__number_of_pixels = 0
        self.__w = 0
        self.__b = 0
        self.__training_accuracy = 0
        self.__testing_accuracy = 0
        self.__true_image_name = ""
        self.__false_image_name = ""
    
    def Create_Dataset(self, Main_File_Name,True_Image_Name, True_File_Name, False_Image_Name, False_File_Name, number_of_pixels = 64, random_state = 42):
        self.__true_image_name = True_Image_Name
        self.__false_image_name = False_Image_Name
        im.Create_Dataset(Main_File_Name, True_Image_Name, True_File_Name, False_Image_Name, False_File_Name, number_of_pixels, random_state)
    
    def LoadDataset(self, Main_File_Name):
        self.__x_array, self.__y_array, self.__classes, self.__true_image_name, self.__false_image_name = dm.load_dataset(Main_File_Name)
        self.__training_samples = self.__x_array.shape[0]
        self.__number_of_pixels = self.__x_array.shape[1]
        return self.__x_array, self.__y_array, self.__classes
    
    def ShowImage(self, index):
        im.Show_Image(index, self.__x_array,self.__y_array, self.__classes)
        
    def TrainModel(self, ratio = 0.8,num_iterations=2000,learning_rate = 0.001, print_cost = False):
        x_array_flattened = dm.flattened_array(self.__x_array, self.__training_samples)
        self.__training_x_array, self.__testing_x_array = dm.train_test_split(self.__training_samples,x_array_flattened, ratio)
        self.__training_y_array, self.__testing_y_array = dm.train_test_split(self.__training_samples, self.__y_array, ratio)
        self.__w,self.__b = dm.initialize_parameters(self.__training_x_array.shape[0])
        params, grads, cost = ct.optimize(self.__w,self.__b,self.__training_x_array,self.__training_y_array,self.__training_samples,num_iterations,learning_rate,print_cost)
        self.__w = params["w"]
        self.__b = params["b"]
        return self.__w,self.__b

    def __predict(self,w,b,x):
        m = x.shape[1]
        classes = []
        
        y_prediction = np.zeros((1,m))
        
        w = w.reshape(x.shape[0],1)
        
        a = ct.sigmoid(np.dot(w.T,x)+b)
        
        for i in range(a.shape[1]):
            if a[0,i] >= 0.52:
                y_prediction[0,i] = 1
                classes.append(self.__true_image_name)
            else:
                y_prediction[0,i] = 0
                classes.append(self.__false_image_name)
        
        return y_prediction, np.array(classes).astype("S")
    
    def TestModel(self):
        self.__training_predictions, self.__training_classes = self.__predict(self.__w, self.__b, self.__training_x_array)
        self.__testing_predictions, self.__testing_classes = self.__predict(self.__w, self.__b, self.__testing_x_array)
        
        testing_accuracy = 100-np.mean(abs(self.__testing_y_array - self.__testing_predictions))*100

        TP = []
        TN = []
        FP = []
        FN = []
        
        for true,pred in zip(self.__testing_y_array[0],self.__testing_predictions[0]):
            if true == 1 and pred == 1:
                TP.append(1)
            if true == 1 and pred == 0:
                FN.append(1)
            if true == 0 and pred == 1:
                FP.append(1)
            if true == 0 and pred == 0:
                TN.append(1)
        
        Precision = (np.sum(TP)/(np.sum(TP) + np.sum(FP)))*100
        Recall = (np.sum(TP)/(np.sum(TP) + np.sum(FN)))*100
        F1Score = (2 * (Precision*Recall)/(Precision + Recall))/100
        Specifity = (np.sum(TN)/(np.sum(TN) + np.sum(FP)))*100
        print("\n📊 Model Evaluation Metrics")
        print("-"*37)
        print(f"✅Accuracy    :{testing_accuracy:.0f}%")
        print(f"🎯Precision   :{Precision:.0f}%")
        print(f"🔁Recall      :{Recall:.0f}%")
        print(f"📐F1Score     :{F1Score:.2f}")
        print(f"🔒Specifity   :{Specifity:.0f}%")  
        print("-"*37)
        
    def ShowImageTrainingPrediction(self,index):
        if index > self.__x_array.shape[0] - self.__training_predictions.shape[1]-1:
            return "Index out of bounds."
        im.Show_Image(index, self.__x_array,self.__training_predictions, self.__training_classes)
        
    def ShowImageTestingPrediction(self,index):
        n = (self.__x_array.shape[0] - self.__testing_predictions.shape[1])
        if index >= self.__testing_predictions.shape[1]-1:
            return "Index out of bounds."
        im.Show_Image(index, self.__x_array[n:],self.__testing_predictions, self.__testing_classes)
        
    def ImagePredict(self, image):
        img_array = im.RGB_Convert(image, self.__number_of_pixels)
        flattened_img_array = dm.flattened_array(img_array, 1)
        prediction, category = self.__predict(self.__w, self.__b,flattened_img_array)
        return f"{category[0].decode('utf-8')}"
    
    def SaveModel(self, Model_Name):
        with h5py.File(Model_Name + ".h5", 'w') as h5f:
            string_dt = h5py.string_dtype(encoding = "utf-8")
            h5f.create_dataset("x_array", data = self.__x_array)
            h5f.create_dataset("training_x_array", data = self.__training_x_array)
            h5f.create_dataset("testing_x_array", data = self.__testing_x_array)
            h5f.create_dataset("y_array", data = self.__y_array)
            h5f.create_dataset("training_y_array", data = self.__training_y_array)
            h5f.create_dataset("testing_y_array", data = self.__testing_y_array)
            h5f.create_dataset("testing_predictions", data = self.__testing_predictions)
            h5f.create_dataset("training_predictions", data = self.__training_predictions)
            h5f.create_dataset("classes", data = self.__classes)
            h5f.create_dataset("training_classes", data = self.__training_classes)
            h5f.create_dataset("testing_classes", data = self.__testing_classes)
            h5f.create_dataset("training_samples", data = self.__training_samples)
            h5f.create_dataset("number_of_pixels", data = self.__number_of_pixels)
            h5f.create_dataset("w", data = self.__w)
            h5f.create_dataset("b", data = self.__b)
            h5f.create_dataset("TrueImageName", data = self.__true_image_name, dtype = string_dt)
            h5f.create_dataset("FalseImageName", data = self.__false_image_name, dtype = string_dt)
            
    def LoadModel(self, Model_Name):
        model = h5py.File(Model_Name + ".h5", "r")
        self.__x_array = model["x_array"][:]
        self.__training_x_array = model["training_x_array"][:]
        self.__testing_x_array = model["testing_x_array"][:]
        self.__y_array = model["y_array"][:]
        self.__training_y_array = model["training_y_array"][:]
        self.__testing_y_array = model["testing_y_array"][:]
        self.__testing_predictions = model["testing_predictions"][:]
        self.__training_predictions = model["training_predictions"][:]
        self.__classes = model["classes"][:]
        self.__training_classes = model["training_classes"][:]
        self.__testing_classes = model["testing_classes"][:]
        self.__training_samples = model["training_samples"][()]
        self.__number_of_pixels = model["number_of_pixels"][()]
        self.__w = model["w"][:]
        self.__b = model["b"][()]
        self.__true_image_name = model["TrueImageName"][()].decode()
        self.__false_image_name = model["FalseImageName"][()].decode()
        print("\nModel uploaded succesfully.")
        
    def ConfusionMatrix(self):
        plt.figure()
        y_test = dm.flattened_array(self.__testing_y_array, self.__testing_y_array.shape[1])[0]
        y_pred = dm.flattened_array(self.__testing_predictions, self.__testing_y_array.shape[1])[0]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues")
        plt.xlabel("Predicted", size = 11)
        plt.ylabel("Actual", size = 11)
        plt.title("Confusion Matrix", size = 18, fontweight = "bold")
        plt.show()
        
            