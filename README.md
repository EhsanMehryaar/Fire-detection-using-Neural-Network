Fire Detection using Neural Networks
==============================

Project Organization
------------

    ├── README.md          <- Project documentation
    │
    ├── models             <- Models included in the project
    │   ├── models.py
    │   
    ├── notebooks          <- Jupyter notebooks of examples of how to use the materials.
    │   ├── Example1.ipynb <- Example of CNN model 
    │   ├── Example2.ipynb <- Example of ANN model
    │   ├── Example2.ipynb <- Example of RCNN model
    │  
    ├── figures            <- Figures generated by the example notebooks
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── google_scraper.py    <- Scrape google for images 
    │   │
    │   ├── utils.py             <- Classifire object and functions used in the project 

Project Documentation
------------

In this project, an easy to use framework for image classification using 3 different NN based models is developed. This framework has been seccusfully applied to scraped images from google. This framework can be usefull for reproducability of the models and are easily modifiable. These models are defined and can be modified in models/models.py. New models can be added. Please note that you have to add the model name to model_generator method in src.utils.model_classifier object. 

Thw whole process of preprocessing the images, data splitting, training and evaluation is done through the image classifier object. Structure of this object is explained below
Examples of the models are given in 3 different Jupyter notebooks in 

- __init__: Initializes the image classifier object with a specified model type (e.g., "cnn", "ann", "rcnn") and a list of class labels.

- _get_paths: Returns a list of paths for each class based on the provided class list.

- _jpg_to_png: Converts all JPG and JPEG images in the class paths to PNG format.

- _create_list_of_image_address: Creates a list of image addresses and labels based on the class paths.

- data_preprocessing: Preprocesses the data by converting JPG images to PNG, shuffling the DataFrame, splitting the data into train and test sets, and creating data generators using ImageDataGenerator from TensorFlow.

- model_generator: Generates the model based on the specified model type.

- summarize_model: Prints a summary of the model's architecture.

- plot_model: Plots and saves a visual representation of the model's architecture.

- fit: Trains the model using the training data and validates it using the validation data.

- plot_accuracy: Plots and saves the accuracy of the model during training and validation.

- plot_loss: Plots and saves the loss of the model during training and validation.

- predict: Generates predictions using the trained model on the test data.

- evaluate: Evaluates the model's performance on the test data and prints the loss and accuracy.

- plot_confusion: Plots and saves a confusion matrix based on the model's predictions and the actual labels.

- plot_result_images: Plots and shows a grid of images from the test data with their actual and predicted labels.

- print_report: Prints a classification report based on the model's predictions and the actual labels.



Structure of the models used in this project are shown below.

This is the CNN model structure

![test](https://github.com/EhsanMehryaar/Fire-detection-using-Neural-Network/blob/main/figures/cnn.png?raw=true)

This is the ANN model structure

![test](https://github.com/EhsanMehryaar/Fire-detection-using-Neural-Network/blob/main/figures/ann.png?raw=true)

This is the RCNN model structure

![test](https://github.com/EhsanMehryaar/Fire-detection-using-Neural-Network/blob/main/figures/rcnn.png?raw=true)

