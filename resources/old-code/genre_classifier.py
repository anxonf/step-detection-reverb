import json
import numpy as np
from sklearn.model_selection import train_test_split
import tf_keras as keras
import matplotlib.pyplot as plt
#import tensorflow.python.keras as keras
#from tensorflow.python.keras.layers import Dense, Flatten
#import keras

DATASET_PATH = "data.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert lists into numpy arrays from the json file
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

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


    # STEP 1: Load data
    inputs, targets = load_data(DATASET_PATH)


    # STEP 2: Split data into train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)


    # STEP 3: Build the network architecture
    #x = keras.Input(shape=(inputs.shape[1], inputs.shape[2]))
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        #keras.layers.Flatten()(x),
       
        # 1st hidden layer
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        
        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
                
        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        
        # output layer
        keras.layers.Dense(10, activation="softmax") # usamos 10 neuronas porque tenemos 10 categorías
    ])


    # STEP 4: Compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.summary()


    # STEP 5: Train network
    history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs= 100, batch_size=32) # batch_size -> compute the gradient on a subset of the data set between 16-128 samples


    # STEP 6: Detect and solve overfitting
    # 2 techniques when overfitting appears
    # - Dropout: elimina aleatoriamente neuronas para cada batch -> evita predicciones al no sabe qué neuronas están activas cada vez
    # en las hidden layers añadimos la función: keras.layers.Dropout(0.3), -> porcentaje de dropout para cada hidden layer
    # - Regularization: penaliza pesos muy elevados de algunas neuronas o caminos -> penaliza error my bajo
    # na las hidden layers que queramos añadimos la opción: kernel_regularizer=keras.regularizers.l2(0.001)
    plot_history(history)

