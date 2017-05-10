import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import pandas as pd

#Load the vectorized question pairs from train and test
print('Loading training and test data...')
train = np.load("../Data/train_vector.npy")
test = np.load("../Data/test_vector.npy")
trainlabels = pd.read_csv('../Data/train.csv')
trainlabels = trainlabels.replace(np.nan, 0, regex=True)

#Separate the vectorized sets into x as features and y as labels
x_train = train[:, :]
y_train = trainlabels.is_duplicate.values
x_test = test[:, :]

#Convert nan values to 0
print('Converting nan values to 0...')
x_train = np.nan_to_num(x_train)
y_train = np.nan_to_num(y_train)
x_test = np.nan_to_num(x_test)

#Initializing hyperparameters
n_features = len(x_train[0])
epochs = 40
batch_size = 1000
dropout_rate = 0.25 #Helps preventing overfitting
n_neurons = 2000
n_hidden_layers = 2
shuffle = True

#Create neural network model
model = Sequential()
model.add(Dense(n_neurons, input_dim=n_features, activation='relu'))
model.add(Dropout(dropout_rate))
for i in range(1, n_hidden_layers):
    #Number of neurons decrease evenly with each layer
    new_n_neurons = int(n_neurons*((n_hidden_layers-i)/n_hidden_layers))
    model.add(Dense(new_n_neurons, activation='relu'))
    model.add(Dropout(dropout_rate))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=shuffle)

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