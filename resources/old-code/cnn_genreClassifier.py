import json
import numpy as np
from sklearn.model_selection import train_test_split
import tf_keras as keras

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

    # for a CNN tensorflow expects a 3D Array (for the depth dimension) -> need to reshape the arrays adding a 3rd dimension = 1
    X_train = X_train[..., np.newaxis] # 4D array -> (num_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):

    # create model -> CNN with 3 convolutional layers followed by max pooling layer
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape)) # 32 kernels, (3, 3) grid size, activation function = relu
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu')) # 64 neurons, type of activation = relu
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


if __name__ == "__main__":

    # STEP 1: Create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # STEP 2: Build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # STEP 3: Compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",  metrics=["accuracy"])
    model.summary()

    # STEP 4: Train the CNN
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # STEP 5: Evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # STEP 6: Make prediction on a sample
    X = X_test[100]
    y = y_test[100]

    predict(model, X, y)
