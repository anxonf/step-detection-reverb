import json
import numpy as np
from sklearn.model_selection import train_test_split
import tf_keras as keras
import matplotlib.pyplot as plt

"""
    MODIFICATIONS FOR THE CNN PROGRAM TO CREATE A RNN-LSTM NETWORK
    FOR GENRE CLASSIFICATIONS
    CODE COPIED FROM:
        cnn_genreClassifier.py
        genreClassifier.py
"""
DATA_PATH = "data.json"

def load_data(data_path):

    """Loads training dataset from json file.

        :param data_path (str); Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets    
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y

def prepare_datasets(test_size, validation_size):
    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train , test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):

    # create model -> RNN-LSTM with 3 convolutional layers followed by max pooling layer
    model = keras.Sequential()

    # create 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True)) # Return_sequencies to True to get sequencies output to pass to the next layer
    model.add(keras.layers.LSTM(64)) # sequence to vector layer, no need for more definition than this

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer softmax
    model.add(keras.layers.Dense(10,  activation='softmax')) # 10 neurons = number of genres 

    return model

def predict(model, X, y):

    X = X[np.newaxis, ...]
    
    # prediction will be a 2D array
    prediction = model.predict(X) # X is a 3D array (130, 13, 1), but predict expects a 4D array > (1, 130, 13, 1)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1) # 1D array that is the index predicted -->> the genre maped with the json

    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))

def plot_history(history):
    fig, axs = plt.subplots(2)

    # create the accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train_accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Eval")
    
    # create the error subplot
    axs[1].plot(history.history["loss"], label="train_error")
    axs[1].plot(history.history["val_loss"], label="test_error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Eval")

    plt.show()

if __name__ == "__main__":

    # STEP 1: Create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2) # 25% testing, 20% del restante 75% se usa como validación

    # STEP 2: Build the RNN-LSTM net
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # STEP 3: Compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",  metrics=["accuracy"]) # for event detection not use accuracy preferr FalsePositives or Precision or Recall
    model.summary()

    # STEP 4: Train the CNN
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    plot_history(history)

    # STEP 5: Evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # STEP 6: Make prediction on a sample
    X = X_test[100]
    y = y_test[100]

    predict(model, X, y)
