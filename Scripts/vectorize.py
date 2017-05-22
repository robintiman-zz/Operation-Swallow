import time
from collections import Counter as mset
import regex as re
from scipy.spatial.distance import cdist
import numpy as np
from nltk.corpus import stopwords
import math


"""
New vectorizer for usage with stop words removal. 
"""
def vectorize(dim, glove, data, is_train):
    q1_arr = data.question1.values
    q2_arr = data.question2.values

    nbr_samples = len(q1_arr)
    totaltime = 0
    percentage = 0
    start = time.time()
    vectorized_words = np.zeros((len(q1_arr), dim * 2))
    not_in_glove = []
    for i in range(0, len(q1_arr)):
        q1_vec = np.zeros((1, dim))
        q2_vec = np.zeros((1, dim))

        # Calculate % complete and estimated time left
        if i % int((nbr_samples / 100)) == 0:
            totaltime += (time.time() - start)
            time_estimate = totaltime / (percentage + 1) * (100 - percentage)
            min = int(time_estimate / 60)
            sec = int(time_estimate - min * 60)
            print("Vectorizing...{0}% complete. Estimated time: {1}:{2}".format(percentage, min, sec, end='\r'))
            start = time.time()
            percentage += 1

        q1 = q1_arr[i].lower()
        q2 = q2_arr[i].lower()

        # Regex separating each word of the questions in a vector
        q1_words = re.findall("\w+", q1)
        q2_words = re.findall("\w+", q2)

        # Remove stopwords
        q1_words, q2_words = remove_stop(q1_words, q2_words)

        for i in range(0, max(len(q1_words), len(q2_words))):
            if i < len(q1_words):
                word1 = q1_words[i]
                try:
                    q1_vec = np.add(q1_vec, glove[word1])
                except KeyError:
                    q1_vec = np.add(q1_vec, hash_word(word1, dim))
            if i < len(q2_words):
                word2 = q2_words[i]
                try:
                    q2_vec = np.add(q2_vec, glove[word2])
                except KeyError:
                    q2_vec = np.add(q2_vec, hash_word(word2, dim))

        vectorized_words[i, :dim] = q1_vec
        vectorized_words[i, dim:] = q2_vec

    print("Writing to file...")
    if is_train:
        np.save("../Data/train_vector", vectorized_words)
    else:
        np.save("../Data/test_vector", vectorized_words)
    print("Done!")

"""
Removes stop words
"""
def remove_stop(q1, q2):
    stop = set(stopwords.words('english'))
    q1_stop = [word for word in q1 if word not in stop]
    q2_stop = [word for word in q2 if word not in stop]
    return q1_stop, q2_stop

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