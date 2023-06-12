## Handwritten Number Recognition with Machine Learning

This project aims to build a machine learning model that can identify handwritten numbers using the MNIST dataset. The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits from 0 to 9.

#### Prerequisites

Before running the code, ensure that you have the following dependencies installed:

numpy
matplotlib
keras
You can install these dependencies using pip:

**pip install numpy matplotlib keras**

#### Code Overview

The provided code performs the following steps:

Imports necessary libraries and modules.
Loads the MNIST dataset using the mnist.load_data() function.
Displays sample images from the dataset.
Preprocesses the image data by normalizing the pixel values and expanding the dimensions.
Converts the target labels into categorical form.
Builds a convolutional neural network (CNN) model using the Keras Sequential API.
Compiles the model with the Adam optimizer and categorical cross-entropy loss.
Defines callbacks for early stopping and model checkpointing.
Trains the model on the training data with a validation split and the defined callbacks.
Getting Started

To run the code and train the model, follow these steps:

Make sure you have the necessary dependencies installed.
Set up your Python environment with the required packages.
Copy and paste the provided code into a Python file, e.g., handwritten_number_recognition.py.
Run the Python file.
Code Snippets

##### Here are some key code snippets from the project:

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape
The code above loads the MNIST dataset, splitting it into training and testing sets. It also displays the shapes of the loaded arrays.

def plot_input_image(i):
plt.imshow(X_train[i], cmap='binary')
plt.title(y_train[i])
plt.show()

for i in range(10):
plot_input_image(i)
The plot_input_image function displays a given image from the training set along with its corresponding label. The for loop then calls this function to show the first 10 images.

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
The code snippet above defines the architecture of the convolutional neural network (CNN) model. It consists of convolutional and pooling layers, followed by a flatten layer, a dropout layer, and a dense layer with softmax activation.

es = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4, verbose=1)
mc = ModelCheckpoint("./bestmodel.h5", monitor="val_acc", verbose=1, save_best_only=True)
cb = [es, mc]
his = model.fit(X_train, y_train, epochs=15, validation_split=0.5, callbacks=cb)
The above code sets up early stopping and model checkpointing callbacks. It then trains the model on the training data for 15 epochs with a validation split and the defined callbacks.

Conclusion
This project demonstrates the process of building and training a machine learning model for handwritten number recognition using the MNIST dataset. By following the provided code and instructions, you should be able to run the code and train your own model.

Feel free to modify the code or experiment with different architectures to improve the model's performance. Good luck with your project!
