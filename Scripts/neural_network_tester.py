import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import model_from_json
import time
from load_files import load_glove
from collections import Counter as mset
import regex as re
from nltk.corpus import stopwords
from scipy.spatial.distance import cdist
import load_files as lf

def vectorize(q1, q2, glove):
    dim = 50
    nbr_features = 15

    q1_vec = np.zeros((1, dim))
    q2_vec = np.zeros((1, dim))
    features = np.zeros((nbr_features))

    q1 = q1.lower()
    q2 = q2.lower()

    # Regex separating each word of the questions in a vector
    q1_words = re.findall("\w+", q1)
    q2_words = re.findall("\w+", q2)

    # Length with and without spaces
    features[0] = abs(len(q1) - len(q2))
    features[1] = abs(len(q1.replace(" ", "")) - len(q2.replace(" ", "")))

    # Remove stopwords
    q1_words, q2_words = remove_stop(q1_words, q2_words)

    # Length without stopwords
    features[2] = abs(sum([len(word) for word in q1_words]) - sum([len(word) for word in q2_words]))

    # Common words
    features[3] = get_common(q1_words, q2_words)

    # GloVe features
    for word in q1_words:
        try:
            q1_vec = np.add(q1_vec, glove[word])
        except KeyError:
            continue
    for word in q2_words:
        try:
            q2_vec = np.add(q2_vec, glove[word])
        except KeyError:
            continue

    # Distance features
    features[4] = cdist(q1_vec, q2_vec, 'euclidean')
    features[5] = cdist(q1_vec, q2_vec, 'cityblock')
    features[6] = cdist(q1_vec, q2_vec, 'cosine')
    features[7] = cdist(q1_vec, q2_vec, 'correlation')
    features[8] = cdist(q1_vec, q2_vec, 'jaccard')
    features[9] = cdist(q1_vec, q2_vec, 'chebyshev')
    features[10] = cdist(q1_vec, q2_vec, 'seuclidean', V=None)
    features[11] = cdist(q1_vec, q2_vec, 'sqeuclidean')
    features[12] = cdist(q1_vec, q2_vec, 'hamming')
    features[13] = cdist(q1_vec, q2_vec, 'canberra')
    features[14] = cdist(q1_vec, q2_vec, 'braycurtis')

    # Normalize the vectorized questions
    # features_norm = np.linalg.norm(features)
    # features = np.divide(features, features_norm)
    # q1_norm = np.linalg.norm(q1_vec)
    # q2_norm = np.linalg.norm(q1_vec)
    # q1_vec = np.divide(q1_vec, q1_norm)
    # q2_vec = np.divide(q2_vec, q2_norm)

    # Convert NaN values to 0
    q1_vec = np.nan_to_num(q1_vec)
    q2_vec = np.nan_to_num(q2_vec)
    features = np.nan_to_num(features)

    vectorized_words = np.zeros((1, dim + nbr_features))
    vectorized_words[0, :dim] = np.abs(np.subtract(q1_vec, q2_vec))
    vectorized_words[0, dim:] = features

    return vectorized_words

"""
Removes stop words
"""
def remove_stop(q1, q2):
    stop = set(stopwords.words('english'))
    q1_stop = [word for word in q1 if word not in stop]
    q2_stop = [word for word in q2 if word not in stop]
    return q1_stop, q2_stop


"""
Returns the number of words not in common
"""
def get_common(q1, q2):
    common = list((mset(q1) & mset(q2)).elements())
    return len(common)

#Load pre-trained model
print("Loading pre-trained model...")
json_file = open("Feed-Forward_Model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("Feed-Forward_Model.h5")

#Load GloVe
glove = lf.load_glove("../Data/glove.6B.50d.txt")

while(True):
    print("Please type two questions.")
    print("Question 1: ", end='')
    q1 = input()
    print("Question 2: ", end='')
    q2 = input()

    vectorized_words = vectorize(q1, q2, glove)

    prediction = model.predict(vectorized_words)[0][0]

    if prediction > 0.5:
        print("Similar with " + str(2*(round(prediction*100, 2) - 50)) + "% confidence")
    else:
        print("Not similar with " + str(2*(50 - round(prediction*100, 2))) + "% confidence")