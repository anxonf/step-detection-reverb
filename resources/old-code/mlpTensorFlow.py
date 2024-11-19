#import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turn off floating-point round-off errors from different computation orders
import keras as ke # keras -> tf library the easies the coding in TF
import numpy as np
from random import random
from sklearn.model_selection import train_test_split

# Program that creates a Neural Network from scratch using Tensor Flow, Steps:
# 1. Build a model
# 2. Compile the model
# 3. Train the model
# 4. Evaluate the model
# 5. Make predictions

def generate_dataset(num_samples, test_size):

    # Data set of the type
    # Inputs: -> array([[0.1,0.2], [0.2,0.2]])
    # Outputs: -> array([[0.3], [0.4]])

    # Dataset

    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])

    # Training set to train the module -> inputs: x_train .. outputs: y_train
    # Test set to evaluate how well the model works -> inputs: x_test .. outputs: y_test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size) # test size = 0.3 -> our test data set is going to be the 30% of the whole dataset samples

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    x_train , x_test, y_train, y_test = generate_dataset(5000, 0.3) # generate the dataset of X samples and only Y% are the test samples

    #print("Inputs for train set x_train: \n {}".format(len(x_train)))
    #print("Outputs for train set y_train: \n {}".format(len(y_train)))
    #print("Inputs for test set x_test: \n {}".format(len(x_test)))
    #print("Outputs for test set y_test: \n {}".format(len(y_test)))
    
    # STEP 1: BUILDING A MODEL WITH TENSOR FLOW USING KERAS of 2 -> 5 -> 1

    model = ke.Sequential([
        ke.layers.Dense(5, input_dim = 2, activation = "sigmoid"),
        ke.layers.Dense(1, activation = "sigmoid")
    ])
    
    # STEP 2: COMPILE THE MODEL
    optimizer = ke.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss="mse") # MSE -> Mean Sqared Error /// loss -> error function

    # STEP 3: TRAIN THE MODEL
    model.fit(x_train, y_train, epochs=100)

    # STEP 4: EVALUATE THE MODEL
    print("\nModel evaluation:")
    model.evaluate(x_test, y_test, verbose=2)

    # STEP 5: MAKE PREDICTIONS
    data = np.array([[0.1,0.2], [0.2,0.2]])
    predictions = model.predict(data)

    print("\nSome predictions:")

    for d, p in zip(data, predictions):
        print("{} + {} = {}".format(d[0], d[1], p[0]))

    