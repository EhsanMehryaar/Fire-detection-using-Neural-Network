# Importing libraries
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D,BatchNormalization,\
                        Permute, TimeDistributed, Bidirectional,GRU, SimpleRNN, LSTM, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.keras.models import Sequential

def cnn_model():
    """CNN model

    Returns:
        model: CNN model
    """
    model = Sequential()
    model.add(Conv2D(32,(3,3), # 32 filters, (3,3) kernel, ReLU activation
                     activation="relu",
                     input_shape=(256,256,3)))
    model.add(BatchNormalization()) # Normalize layer activations
    model.add(MaxPooling2D((2,2))) # Max pooling with (2,2) pool size

    model.add(Conv2D(64,(3,3), # 64 filters, (3,3) kernel, ReLU activation, padding
                     activation="relu",
                     padding="same"))
    model.add(Dropout(0.2)) # Dropout regularization to prevent overfitting
    model.add(MaxPooling2D((2,2))) # Max pooling with (2,2) pool size

    model.add(Conv2D(128,(3,3), # 128 filters, (3,3) kernel, ReLU activation, padding
                    activation="relu",
                    padding="same"))
    model.add(Dropout(0.5)) # Dropout regularization to prevent overfitting
    model.add(MaxPooling2D((2,2))) # Max pooling with (2,2) pool size

    model.add(Flatten()) # Flatten the output from previous layer
    model.add(Dense(256,activation="relu")) # Fully connected layer with 256 units and ReLU activation
    model.add(Dense(128,activation="relu")) # Fully connected layer with 128 units and ReLU activation
    model.add(Dropout(0.5)) # Dropout regularization to prevent overfitting
    model.add(Dense(2,activation="sigmoid")) # Output layer with 2 units and sigmoid activation for binary classification


    return model

def ann_model():
    """ANN model

    Returns:
        model: ANN model
    """
    model = tf.keras.models.Sequential([
    # inputs 
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255), # Normalize pixel values between 0 and 1
    tf.keras.layers.Flatten(input_shape=(256,)), # Flatten the input
    # hiddens layers
    tf.keras.layers.Dense(128, activation='relu'), # Fully connected layer with 128 units and ReLU activation
        tf.keras.layers.Dense(64, activation='relu'), # Fully connected layer with 64 units and ReLU activation
    tf.keras.layers.Dropout(0.2), # Dropout regularization to prevent overfitting
    # output layer
    tf.keras.layers.Dense(2,activation="softmax") # Output layer with 2 units and softmax activation for binary classification
    ])
    
    return model
    
def rcnn_model():
    """RCNN model

    Returns:
        model: RCNN model
    """
    model = Sequential()

    model.add(Conv2D(12,(3,3), # 12 filters, (3,3) kernel, ReLU activation
                     activation="relu",
                     input_shape=(256,256,3)))
    model.add(BatchNormalization()) # Normalize layer activations
    model.add(MaxPooling2D((2,2))) # Max pooling with (2,2) pool size

    model.add(Conv2D(24,(3,3), # 24 filters, (3,3) kernel, ReLU activation
                    activation="relu"))
    model.add(Dropout(0.2)) # Dropout regularization to prevent overfitting
    model.add(MaxPooling2D((2,2))) # Max pooling with (2,2) pool size

    model.add(TimeDistributed(Flatten())) # Time-distributed flatten layer
    model.add(Bidirectional(LSTM(32, # Bidirectional LSTM layer with 32 units, dropout, and recurrent dropout
                                    return_sequences=True,
                                    dropout=0.5,
                                    recurrent_dropout=0.5)))
    model.add(Bidirectional(GRU(32, # Bidirectional GRU layer with 32 units, dropout, and recurrent dropout
                                    return_sequences=True,
                                    dropout=0.5,
                                    recurrent_dropout=0.5)))

    model.add(Flatten()) # Flatten the output from previous layer
    model.add(Dense(256,activation="relu"))  # Fully connected layer with 256 units and ReLU activation
    model.add(Dropout(0.5)) # Dropout regularization to prevent overfitting
    model.add(Dense(2,activation="softmax")) # Output layer with 2 units and softmax activation for binary classification

    return model