from keras.models import Sequential
from keras.layers import LSTM
import pandas as pd
import math
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Merge
from keras.layers import Conv1D, MaxPooling1D, Embedding


def tokenize(values):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(values)
    sequences = tokenizer.texts_to_sequences(values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return data, word_index

def load_embedding(load_from_file=True):
    if not load_from_file:
        embeddings_index = {}
        f = open(GLOVE_DIR)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        np.save('../Data/embedding', embeddings_index)
        f.close()
    else:
        embeddings_index = np.load("../Data/embedding.npy").item()
    return embeddings_index


def build_embedding_matrix(word_index):
    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index))
    print(len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i < 0:
            print("ksjgs")
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i - 1] = embedding_vector
    return embedding_matrix, num_words

GLOVE_DIR = '../Data/glove.6B.50d.txt'
DATA_DIR = '../Data/train.csv'
MAX_SEQUENCE_LENGTH = 120
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2

print('Indexing word vectors.')
embeddings_index = load_embedding(False)
print('Found %s word vectors.' % len(embeddings_index))

print('Reading data.')
dataset = pd.read_csv('../Data/train.csv', dtype=str)
labels = dataset.is_duplicate.values

# finally, vectorize the text samples into a 2D integer tensor
q1_data, word_index1 = tokenize([str(word) for word in dataset.question1.values])
q2_data, word_index2 = tokenize([str(word) for word in dataset.question2.values])
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
print("Splitting data into a training set and a validation set.")
indices = np.arange(q1_data.shape[0]) # q1_data and q2_data should be the same length
np.random.shuffle(indices)
q1_data = q1_data[indices]
q2_data = q2_data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * q1_data.shape[0])

x_train1 = q1_data[:-num_validation_samples]
x_train2 = q2_data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val1 = q1_data[-num_validation_samples:]
x_val2 = q2_data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrices.')
embedding_matrix1, num_words1 = build_embedding_matrix(word_index1) # load pre-trained word embeddings into an Embedding layer
embedding_matrix2, num_words2 = build_embedding_matrix(word_index2)

print(x_train1.shape)
# note that we set trainable = False so as to keep the embeddings fixed
branch1 = Sequential()
branch1.add(Embedding(num_words1, EMBEDDING_DIM, weights=[embedding_matrix1],
                      input_length=MAX_SEQUENCE_LENGTH, trainable=False, input_shape=(MAX_SEQUENCE_LENGTH, )))
branch1.add(LSTM(EMBEDDING_DIM))

branch2 = Sequential()
branch2.add(Embedding(num_words2, EMBEDDING_DIM, weights=[embedding_matrix2],
                      input_length=MAX_SEQUENCE_LENGTH, trainable=False, input_shape=(MAX_SEQUENCE_LENGTH, )))
branch2.add(LSTM(EMBEDDING_DIM))

model = Sequential()
model.add(Merge([branch1, branch2], mode='concat'))
# model.add(LSTM(EMBEDDING_DIM, recurrent_activation='relu', recurrent_dropout=0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit([x_train1, x_train2], y_train, epochs=10, batch_size=128, validation_data=([x_val1, x_val2], y_val))

model_json = model.to_json()
with open("../Data/LSTM.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("../Data/LSTM.h5")
print("Saved model to disk")

# Test our trained model on the test set
print('Applying model to test set...')
x_test = np.load("../Data/test_vector.npy")
predictions = model.predict(x_test)

# Write the test predictions to submission file
print("Writing test results to file...")
f = open("keras_predictions", 'w')
f.write("test_id,is_duplicate\n")
counter = 0
for val in np.nditer(predictions):
    f.write(str(counter) + "," + str(val) + "\n")
    counter += 1
f.close()
print("Done!")