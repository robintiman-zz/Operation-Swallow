import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import model_from_json
import pandas as pd

#Load the vectorized question pairs from train and test
print('Loading training and test data...')
x_train = np.load("../Data/train_vector.npy")
trainlabels = pd.read_csv('../Data/train.csv')
trainlabels = trainlabels.replace(np.nan, 0, regex=True)
y_train = trainlabels.is_duplicate.values
x_test = np.load("../Data/test_vector.npy")

#Convert nan values to 0
print('Converting nan values to 0...')
x_train = np.nan_to_num(x_train)
y_train = np.nan_to_num(y_train)
x_test = np.nan_to_num(x_test)

# Hyperparameters
n_features = len(x_train[0])
epochs = 30
batch_size = 1024
dropout_rate = 0.4 # Helps preventing overfitting
n_neurons = 2000
n_hidden_layers = 2
shuffle = True
train = True

model = None
if train:
    # Create neural network model, save to disk when done
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=n_features, activation='relu'))
    model.add(Dropout(dropout_rate))
    for i in range(1, n_hidden_layers):
        #Number of neurons decrease evenly with each layer
        n_neurons = int(n_neurons/2)

        model.add(Dense(n_neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(lr=0.01, loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Train the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=shuffle)

    #Save model
    model_json = model.to_json()
    with open("Feed-Forward_Model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("Feed-Forward_Model.h5")
    print("Saved model to disk")

if not train:
    #Load pre-trained model
    print("Loading pre-trained model...")
    json_file = open("Feed-Forward_Model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("Feed-Forward_Model.h5")

    #Test our trained model on the test set
    print('Applying model to test set...')
    predictions = model.predict(x_test)

    #Write the test predictions to submission file
    print("Writing test results to file...")
    f = open("keras_predictions", 'w')
    f.write("test_id,is_duplicate\n")
    counter = 0
    for val in np.nditer(predictions):
        f.write(str(counter) + "," + str(val) + "\n")
        counter += 1
    f.close()
    print("Done!")