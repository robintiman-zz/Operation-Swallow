import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import pandas as pd
from keras.models import model_from_json

#Load the vectorized question pairs from train and test
print('Loading training and test data...')
x_train = np.load("/media/calle/SSD/OperationSwallowData/LSTM_train_vector.npy")
x_train = x_train[:int(len(x_train)/2), :, :]
trainlabels = pd.read_csv('../Data/train.csv')
trainlabels = trainlabels.replace(np.nan, 0, regex=True)
y_train = trainlabels.is_duplicate.values[:len(x_train)]
#x_test = np.load("/media/calle/7E549DAA549D6625/OperationSwallowData/LSTM_test_vector.npy")

#Hyperparameters
n_features = len(x_train[0][0])
max_n_words = len(x_train[0]) #Timesteps
epochs = 30
batch_size = 1000
dropout_rate = 0.5 #Helps preventing overfitting
n_neurons = n_features
shuffle = True
train = True
test = False

model = None
if train:
    #Create LSTM neural network
    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(max_n_words,  n_features), activation='relu', return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(n_neurons, input_shape=(max_n_words, n_features), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Train the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=shuffle)

    #Save model
    model_json = model.to_json()
    with open("LSTM_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("LSTM_model.h5")
    print("Saved model to disk")

if test:
    #Load pre-trained model
    print("Loading pre-trained model...")
    json_file = open("LSTM_model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("LSTM_model.h5")

    #Test our trained model on the test set
    print('Applying model to test set...')
    #predictions = model.predict(x_test)

    #Write the test predictions to submission file
    #print("Writing test results to file...")
    #f = open("keras_predictions", 'w')
    #f.write("test_id,is_duplicate\n")
    #counter = 0
    #for val in np.nditer(predictions):
    #    f.write(str(counter) + "," + str(val) + "\n")
    #    counter += 1
    #f.close()
    #print("Done!")