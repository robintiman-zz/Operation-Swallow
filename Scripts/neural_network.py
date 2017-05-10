import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import *

#Load the vectorized question pairs from train and test
train = np.load("train_vector.npy")
test = np.load("test_vector.npy")

#Separate the vectorized sets into x as features and y as labels
x_train = train[:, 4:]
y_train = train[:, 0]
x_test = test[1:, 1:]
#y_test = train[:, 4:]

#Initializing hyperparameters
n_features = len(x_train[0])
epochs = 500
batch_size = 200
shuffle=True
dropout_rate = 0.2 #Helps preventing overfitting

#Create neural network model
model = Sequential()
model.add(Dense(n_features, input_dim=n_features, activation='softplus'))
model.add(Dropout(dropout_rate))
model.add(Dense(n_features, activation='softplus'))
model.add(Dropout(dropout_rate))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_logarithmic_error', optimizer='Adam', metrics=['accuracy'])

#Train the model
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

#Test our trained model on the test set
predictions = model.predict(x_test)

#Write the test predictions to submission file
print("Writing to file...")
f = open("keras_predictions", 'w')
f.write("test_id,is_duplicate\n")
counter = 0
for val in np.nditer(predictions):
    f.write(str(counter) + "," + str(val) + "\n")
    counter += 1
f.close()
print("Done!")