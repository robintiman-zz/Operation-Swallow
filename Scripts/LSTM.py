from keras.models import Sequential
from keras.layers import LSTM
import pandas as pd
import math
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding

"""
When a word is not present in GloVe, we hash it instead
"""
def hash_word(word, dim):
    hashed_word = np.zeros((1, dim))
    h = sum(bytearray(word,'utf8'))/10000
    for i in range(0, dim):
        f = lambda x: 1 - 1/(math.exp(2*(x/dim + h)) + 1)
        hashed_word[0, i] = f(i)
    return hashed_word

# fix random seed for reproducibility
np.random.seed(7)

GLOVE_DIR = '../Data/glove.6B.50d.txt'
TEXT_DATA_DIR = '../Data/train.csv'
MAX_SEQUENCE_LENGTH = 120
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2

print('Indexing word vectors.')

embeddings_index = {}
f = open(GLOVE_DIR)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

print('Reading data')
dataset = pd.read_csv('../Data/train.csv', dtype=str)
labels = dataset.is_duplicate.values
question_list = []
for i in range(0, len(labels)):
    q1 = dataset.question1.values[i]
    q2 = dataset.question2.values[i]
    question_list.append(str(q1)+ " " + str(q2))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer()
tokenizer.fit_on_texts(question_list)
sequences = tokenizer.texts_to_sequences(question_list)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = hash_word(word, EMBEDDING_DIM)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))

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