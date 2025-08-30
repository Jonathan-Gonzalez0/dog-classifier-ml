# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 14:14:40 2025

@author: Jonathan Gonzalez Rodriguez, jonathan.gonzalez57@upr.edu

Image Classifier Class

This Python program defines a class that implements a linear regression model 
for image classification and prediction tasks. It includes functionality for 
training the model, making predictions, evaluating performance using metrics 
such as accuracy, F1 score, and confusion matrix, and visualizing results with 
tools like Seaborn and Matplotlib.

Last Update: 8/10/2025
"""
import image_manager as im
import dataset_manager as dm
import class_tools as ct
import numpy as np
import h5py
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ImageRegressor:
    """
    A linear regression model for image based prediction tasks.

    This class allows you to train, evaluate, and visualize the performance
    of a regression model using image data as input features.
    """
    
    def __init__(self):
        """
        Initializes all internal variables used by the model.
        
        Returns
        -------
        None.

        """
        self.__x_array = []
        self.__training_x_array = []
        self.__testing_x_array = []
        self.__y_array = []
        self.__training_y_array = []
        self.__testing_y_array = []
        self.__testing_predictions = []
        self.__training_predictions = []
        self.__classes_names = []
        self.__training_classes_names = [] 
        self.__testing_classes_names = []
        self.__w = 0
        self.__b = 0
        self.__training_accuracy = 0
        self.__testing_accuracy = 0
        self.__true_image_name = ""
        self.__false_Image_Name = ""
    
    def create_dataset(self, main_file_name,true_image_name, true_file_name, false_Image_Name, false_file_name, number_of_pixels = 64, random_state = 42):
        """
        Wrapper function that calls Create_Dataset() from the ImageManager 
        module.

        Converts images to RGB-normalized values and stores them in an .h5 
        file, while also organizing, randomizing, and constructing the training 
        dataset.

        Parameters
        ----------
        main_file_name : str
            Variable used as a name to store the RGB converted training set in a 
            .h5 file.
        true_image_name : str
            Name given to True images.
        true_file_name : str
            Folder containing the true images.
        false_Image_Name : str
            Name given to false images.
        false_file_name : str
            Folder containing the false images.
        number_of_pixels : int, optional
            User defined number of pixels per image for RGB conversion. 
            The default is 64.
        random_state : int, optional
            Seed value used to ensure reproducibility of random operations. 
            The default is 42.

        Returns
        -------
        None.

        """
        self.__true_image_name = true_image_name 
        self.__false_Image_Name = false_Image_Name
        im.create_rgb_dataset(main_file_name, true_image_name, true_file_name, false_Image_Name, false_file_name, number_of_pixels, random_state)
    
    def load_dataset(self, main_file_name):
        """
        Loads the dataset that has been converted to RGB 
        format.

        Parameters
        ----------
        main_file_name : str
            File containing dataset that has been converted to RGB format.

        Returns
        -------
        np.ndarray
            NumPy array contaning images converted to RGB.
        np.ndarray
            Binary NumPy array containing 0 and 1 values (1 = positive class, 
            0 = negative class)
        np.ndarray
            NumPy array containing image classification labels.
        """
        dataset = h5py.File(main_file_name, 'r')
        self.__x_array = np.array(dataset["set_x"][:])
        self.__y_array = np.array(dataset["set_y"][:])
        self.__classes_names = np.array(dataset["list_classes_names"][:])
        self.__true_image_name = dataset["true_name"][()].decode()
        self.__false_Image_Name = dataset["false_name"][()].decode()
        
        print("Dataset succesfully loaded.")
        
        return self.__x_array, self.__y_array, self.__classes_names
    
    def show_image(self, index):
        """
        Wrapper function that calls show_image() from ImageManager. 
        module. 

        Shows image from the internal dataset.
        
        Parameters
        ----------
        index : int
            Index indicating which image to visualize from the dataset.

        Returns
        -------
        None.
        """
        im.show_image(index, self.__x_array,self.__y_array, self.__classes_names)
        
    def train_model(self, ratio = 0.8,epochs=2000,learning_rate = 0.001, print_cost = False):
        """
        Function used to train the model on given training data.
        
        Parameters
        ----------
        ratio : float, optional
            Ratio of training vs testing data (default is 0.8).
            For example, 0.8 means 80% training and 20% testing.
        epochs : int, optional
            Number of desired iterations to train the model on (default is 2000). 
        learning_rate : float, optional
            The desired rate of learning for the model (default is 0.001).
            Controls how much the model weights are updated during training.
            Smaller values lead to slower learning; larger values can speed it up but risk overshooting.
        print_cost: bool, optional
            Whether to print the cost every 100 iterations (default is False).
            
        Returns
        -------
        w : np.ndarray
            NumPy array containing all of the computed weights after training.
        b : float
            Bias of the model.
        """
        print("\nTraining model.")
        
        samples = self.__x_array.shape[0]
        
        x_array_flattened = dm.flattened_array(self.__x_array, samples)
        
        self.__training_x_array, self.__testing_x_array = dm.train_test_split(samples,x_array_flattened, ratio)
        self.__training_y_array, self.__testing_y_array = dm.train_test_split(samples, self.__y_array, ratio)
        self.__w,self.__b = ct.initialize_parameters(x_array_flattened.shape[0])
        params, grads, cost = ct.optimize(self.__w,self.__b,self.__training_x_array,self.__training_y_array,samples,epochs,learning_rate,print_cost)
        self.__w = params["w"]
        self.__b = params["b"]
        return self.__w,self.__b

    def __predict(self,w,b,x):
        """
        This function is intended to be hiddened.
        
        Generate a NumPy array with predictions based on an input x_array and 
        given weights.
        
        Parameters
        ----------
        w : np.ndarray
            Array containing model weights used to compute predictions.
        b : float
            Bias of the model, added to the weighted sum to shift the output.
        x : np.array
            Input array of shape (1, m), where m is the number of samples.

        Returns
        -------
        y_predictions : np.ndarray
            Predictions as a NumPy array of shape (1, m).
        classes_names : np.ndarray
            Array with labels of predictions.
        """
        m = x.shape[1] # Number of samples.
        
        classes_names = []
        
        y_prediction = np.zeros((1,m))
        
        w = w.reshape(x.shape[0],1)
        
        a = ct.sigmoid(np.dot(w.T,x)+b) # Linear equation evaluated in the sigmoid function.
        
        for value in range(a.shape[1]):
            if a[0,value] >= 0.4:
                y_prediction[0,value] = 1
                classes_names.append(self.__true_image_name)
            else:
                y_prediction[0,value] = 0
                classes_names.append(self.__false_Image_Name)
                
        classes_names = np.array(classes_names).astype("S")
        
        return y_prediction, classes_names
    
    def test_model(self):
        """
        Evaluate the model performance based on predictions and true labels.
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.__training_predictions, self.__training_classes_names = self.__predict(self.__w, self.__b, self.__training_x_array)
        self.__testing_predictions, self.__testing_classes_names = self.__predict(self.__w, self.__b, self.__testing_x_array)
        
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
        
        precision = (np.sum(TP)/(np.sum(TP) + np.sum(FP)))
        recall = (np.sum(TP)/(np.sum(TP) + np.sum(FN)))
        F1Score = (2 * (precision*recall)/(precision + recall))
        specifity = (np.sum(TN)/(np.sum(TN) + np.sum(FP)))*100
        print("\n📊 Model Evaluation Metrics")
        print("-"*37)
        print(f"✅Accuracy    :{testing_accuracy:.0f}%")
        print(f"🎯Precision   :{precision*100:.0f}%")
        print(f"🔁Recall      :{recall*100:.0f}%")
        print(f"📐F1Score     :{F1Score:.2f}")
        print(f"🔒Specifity   :{specifity:.0f}%")  
        print("-"*37)
        
    def show_image_training_prediction(self,index):
        """
        Generates an image with prediction of the training array.
        
        Parameters
        ----------
        index : int
            Index of the training array to generate the image for.

        Returns
        -------
        None.
        """     
        if index > self.__x_array.shape[0] - self.__training_predictions.shape[1]-1:
            return "Index out of bounds."
        im.show_image(index, self.__x_array,self.__training_predictions, self.__training_classes_names)
        
    def show_image_testing_prediction(self,index):
        """
        Generates an image with prediction of the testing array.
        
        Parameters
        ----------
        index : int
            Index of the testing array to generate the image for.

        Returns
        -------
        None.
        """  
        n = (self.__x_array.shape[0] - self.__testing_predictions.shape[1]) # Number of training predictions. 
        if index >= self.__testing_predictions.shape[1]-1:
            return "Index out of bounds."
        
        x_array_testing_data = self.__x_array[n:]
        
        im.show_image(index, x_array_testing_data,self.__testing_predictions, self.__testing_classes_names)
        
    def image_predict(self, image):
        """
        Returns the prediction of the given image.
        
        Parameters
        ----------
        image : PIL.Image.Image
            Image used to make prediction.

        Returns
        -------
        prediction_label : str
            Predicted label for the input image.
        """
        img_array = im.rgb_convert(image, self.__x_array.shape[1])
        
        flattened_img_array = dm.flattened_array(img_array, 1) # Only one image is converted.
        
        prediction, category = self.__predict(self.__w, self.__b,flattened_img_array)
        
        prediction_label = str(category[0].decode('utf-8'))
        
        return prediction_label
    
    def save_model(self, model_name):
        """
        Saves the trained model to a file for later use.
        
        Parameters
        ----------
        model_name : Str
            Name of the file to save the model. The file is saved in the same 
            directory as the class file.
            
        Returns
        -------
        None.
        """
        with h5py.File(model_name + ".h5", 'w') as h5f:
            string_dt = h5py.string_dtype(encoding = "utf-8")
            h5f.create_dataset("x_array", data = self.__x_array)
            h5f.create_dataset("training_x_array", data = self.__training_x_array)
            h5f.create_dataset("testing_x_array", data = self.__testing_x_array)
            h5f.create_dataset("y_array", data = self.__y_array)
            h5f.create_dataset("training_y_array", data = self.__training_y_array)
            h5f.create_dataset("testing_y_array", data = self.__testing_y_array)
            h5f.create_dataset("testing_predictions", data = self.__testing_predictions)
            h5f.create_dataset("training_predictions", data = self.__training_predictions)
            h5f.create_dataset("classes_names", data = self.__classes_names)
            h5f.create_dataset("training_classes_names", data = self.__training_classes_names)
            h5f.create_dataset("testing_classes_names", data = self.__testing_classes_names)
            h5f.create_dataset("w", data = self.__w)
            h5f.create_dataset("b", data = self.__b)
            h5f.create_dataset("true_image_name", data = self.__true_image_name, dtype = string_dt)
            h5f.create_dataset("false_image_name", data = self.__false_Image_Name, dtype = string_dt)
            
    def load_model(self, model_name):
        """
        Loads a saved model from the specified file.
        
        Parameters
        ----------
        model_name : Str
            Name of the file containing the saved model.
            
        Returns
        -------
        None.
        """
        model = h5py.File(model_name + ".h5", "r")
        self.__x_array = model["x_array"][:]
        self.__training_x_array = model["training_x_array"][:]
        self.__testing_x_array = model["testing_x_array"][:]
        self.__y_array = model["y_array"][:]
        self.__training_y_array = model["training_y_array"][:]
        self.__testing_y_array = model["testing_y_array"][:]
        self.__testing_predictions = model["testing_predictions"][:]
        self.__training_predictions = model["training_predictions"][:]
        self.__classes_names = model["classes_names"][:]
        self.__training_classes_names = model["training_classes_names"][:]
        self.__testing_classes_names = model["testing_classes_names"][:]
        self.__w = model["w"][:]
        self.__b = model["b"][()]
        self.__true_image_name = model["true_image_name"][()].decode()
        self.__false_Image_Name = model["false_image_name"][()].decode()
        print("\nModel uploaded succesfully.")
        
    def confusion_matrix(self):
        """
        Creates a confusion matrix to evaluate classification performance.
        
        Parameters
        ----------
        None.
            
        Returns
        -------
        None.
        """
        plt.figure()
        cm = confusion_matrix(self.__testing_y_array[0][:], self.__testing_predictions[0][:])
        sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues")
        plt.xlabel("Predicted", size = 11)
        plt.ylabel("Actual", size = 11)
        plt.title("Confusion Matrix", size = 18, fontweight = "bold")
        plt.show()
        
            