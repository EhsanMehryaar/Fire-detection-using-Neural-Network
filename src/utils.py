import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import tensorflow as tf


def jpg_to_png(path: str):
    """Converts all JPG and JPEG images in a given folder to PNG format.

    Args:
        path (str): Folder name inside the data folder.
    """    
    import os
    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img = Image.open(os.path.join(path, filename))
            new_filename = filename.split('.')[0] + '.png'
            img.save(os.path.join(path, new_filename))
            os.remove(os.path.join(path, filename))
            
def plot_bar_chart (df: pd.DataFrame):
    """Uses seaborn library to create bar chart based on a provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a column named label to be plotted.
    """    
    plt.figure(figsize=(6, 6))
    sns.countplot(x='label', data=df)
    plt.show()

def plot_pie_chart (df: pd.DataFrame):
    """Uses matplotlib library to create pie chart based on a provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a column named label to be plotted.
    """    
    plt.figure(figsize=(6, 6))
    plt.pie(df['label'].value_counts(), labels=df['label'].value_counts().index, autopct='%1.1f%%')
    plt.show()
    
def plot_single_image(df: pd.DataFrame, img_number: int):
    """Uses matplotlib and cv2 libraries to plot a single image with its' respective label.

    Args:
        df (pd.DataFrame): DataFrame containing a column named label and address to be plotted.
        img_number (int): Number of the image to be plotted.
    """    
    plt.figure(figsize=(6,6))
    x = cv2.imread(df["address"][img_number])
    plt.imshow(x)
    plt.xlabel(f"Image address is {df['address'][img_number]}")
    plt.title(df["label"][img_number])
    plt.show()
    
def plot_multi_images(df: pd.DataFrame, img_number: int):
    """Uses matplotlib and cv2 libraries to plot a multiple images with their respective labels.

    Args:
        df (pd.DataFrame): DataFrame containing a column named label and address to be plotted.
        img_number (int): Number of the images to be plotted.
    """    
    plt.figure(figsize=(20, 20))
    for i in range(5 * (img_number//5)):
        plt.subplot(5, img_number//5, i+1)
        x = cv2.imread(df["address"][i])
        plt.imshow(x)
        plt.xlabel(f"Image address is {df['address'][i]}")
        plt.title(df["label"][i])
    plt.show()
            
class image_classifier:
    
    def __init__(self, model_type: str, class_list: list) -> None:
        """Initializes the image classifier object with a specified model type (e.g., "cnn", "ann", "rcnn") and a list of class labels.

        Args:
            model_type (str): Type of the model (CNN, ANN, RCNN).
            class_list (list): Name of the folders in data folder which are the same as class labels.
        """        
        self.model_type = model_type
        self.class_list = class_list
        self.paths = self._get_paths()
        self.addresses, self.labels = self._create_list_of_image_address()
        self.call_back = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=5,mode="min")
    
    def _get_paths(self) -> list:
        """Returns a list of paths for each class based on the provided class list.

        Returns:
            list: A list of Paths.
        """                
        paths = []
        for class_name in self.class_list:
            path = os.path.join('data', class_name)
            paths.append(path)
        return paths
    
    def _jpg_to_png (self) -> None:
        """Converts all JPG and JPEG images in the class paths to PNG format.
        """        
        for path in self.paths:
            jpg_to_png(path)
            
    def _create_list_of_image_address(self) -> list:
        """Creates a list of image addresses and labels based on the class paths.

        Returns:
            address_list: Addresses of the images.
            label_list: Labels of the images.
        """        
        label_list = []
        address_list = []
        for path in self.paths:
            for filename in os.listdir(path):
                if filename.endswith('.png'):
                    label_list.append(path.split('/')[1])
                    address_list.append(os.path.join(path, filename))
        return address_list, label_list
    
    def data_preprocessing (self) -> None:
        """Preprocesses the data by converting JPG images to PNG, shuffling the DataFrame, 
           splitting the data into train and test sets, and creating data generators using 
           ImageDataGenerator from TensorFlow.
        """
        
        # Import necessary modules        
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # convert any jpg image to png
        self._jpg_to_png()
        
        # Create DataFrame from image addresses and labels
        self.df = pd.DataFrame({'address': self.addresses, 'label': self.labels})
        
        # Shuffle the DataFrame
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        
        # Create ImageDataGenerator for data augmentation and rescaling
        train_generator = ImageDataGenerator(rescale=1./255,
                                             shear_range=0.3,
                                             zoom_range=0.2,
                                             brightness_range=[0.2,0.9],
                                             rotation_range=30,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             fill_mode="nearest",
                                             validation_split=0.1)
        
        # Create ImageDataGenerator for pixel value rescaling
        test_generator = ImageDataGenerator(rescale=1./255)
        
        # Split the shuffled DataFrame into train and test DataFrames
        self.train_df, self.test_df = train_test_split(self.df,train_size=0.8,random_state=42,shuffle=True)
        
        # Create a LabelEncoder instance to encode class labels
        encode = LabelEncoder()
        
        # Encode the labels in the test DataFrame
        encoded_test_class = encode.fit_transform(self.test_df["label"])
        
        # Create ImageDataGenerator for data augmentation and rescaling
        self.train_image_set = train_generator.flow_from_dataframe(dataframe=self.train_df,
                                                                   x_col="address",
                                                                   y_col="label",
                                                                   color_mode="rgb",
                                                                   class_mode="categorical",
                                                                   batch_size=32,
                                                                   subset="training")
        
        self.val_image_set = train_generator.flow_from_dataframe(dataframe=self.train_df,
                                                                 x_col="address",
                                                                 y_col="label",
                                                                 color_mode="rgb",
                                                                 class_mode="categorical",
                                                                 batch_size=32,
                                                                 subset="validation")
        self.test_image_set = test_generator.flow_from_dataframe(dataframe=self.test_df,
                                                                 x_col="address",
                                                                 y_col="label",
                                                                 color_mode="rgb",
                                                                 class_mode="categorical",
                                                                 batch_size=32)
    def model_generator(self) -> None:
        """Generates the model based on the specified model type.

        Raises:
            ValueError: If the model type is not valid.
        """        
        if self.model_type == "cnn":
            from models.models import cnn_model
            self.model = cnn_model()
        elif self.model_type == "ann":
            from models.models import ann_model
            self.model = ann_model()
        elif self.model_type == "rcnn":
            from models.models import rcnn_model
            self.model = rcnn_model()
        else:
            raise ValueError("Model type is not valid")
        self.model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])
        
        
    def summarize_model(self) -> None:
        """Prints a summary of the model's architecture.
        """        
        print(self.model.summary())
    
    def plot_model(self) -> None:
        """Plots and saves a visual representation of the model's architecture.
        """        
        from keras.utils import plot_model
        plot_model(self.model, to_file=self.model_type + '.png', show_shapes=True, show_dtype=True, show_layer_names=True, )
        plt.show()
        
    def fit(self) -> None:
        """Trains the model using the training data and validates it using the validation data.
        """        
        self.history = self.model.fit(self.train_image_set,
                           validation_data=self.val_image_set,
                           callbacks=self.call_back,
                           epochs=50)
        
    def plot_accuracy(self) -> None:
        """Plots and saves the accuracy of the model during training and validation.
        """        
        plt.figure(figsize=(4,4))
        plt.plot(self.history.history["accuracy"])
        plt.plot(self.history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Train","Validation"],loc="upper left")
        plt.savefig(os.path.join('figures', self.model_type + '_accuracy.png'))
        plt.show()
    
    def plot_loss(self) -> None:
        """Plots and saves the loss of the model during training and validation.
        """        
        plt.figure(figsize=(4,4))
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Train","Validation"],loc="upper left")
        plt.savefig(os.path.join('figures', self.model_type + '_loss.png'))
        plt.show()
        
    def predict(self) -> None:
        """Generates predictions using the trained model on the test data.
        """        
        self.predictions = self.model.predict(self.test_image_set)
        self.predictions = self.predictions.argmax(axis=-1)
        
    def evaluate(self) -> None:
        """Evaluates the model's performance on the test data and prints the loss and accuracy.
        """        
        evaluation = self.model.evaluate(self.test_image_set)
        print ("Loss is equal to " + "%.4f" % evaluation[0])
        print("Accuracy is equal to " + "%.2f" % evaluation[1])
        
    def plot_confusion(self) -> None:
        """Plots and saves a confusion matrix based on the model's predictions and the actual labels.
        """        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.test_image_set.classes,self.predictions)
        plt.figure(figsize=(4,4))
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join('figures', self.model_type + '_confusion_matrix.png'))
        plt.show()
        
    def plot_result_images(self) -> None:
        """Plots and shows a grid of images from the test data with their actual and predicted labels.
        """        
        fig, axes = plt.subplots(nrows=8,
                                 ncols=8,
                                 figsize=(20, 20),
                                 subplot_kw={'xticks': [], 'yticks': []})
        for i, ax in enumerate(axes.flat):
            ax.imshow(cv2.imread(self.test_df["address"].iloc[i]))
            ax.set_title(f"actual:{self.test_df.label.iloc[i]}\n prediction:{self.predictions[i]}")
        plt.tight_layout()
        plt.show()
        
    def print_report(self) -> None:
        """Prints a classification report based on the model's predictions and the actual labels.
        """        
        from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
        print(classification_report(self.test_image_set.classes,self.predictions))
        
