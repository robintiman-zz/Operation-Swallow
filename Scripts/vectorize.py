import numpy as np
import time
from load_files import load_glove
from collections import Counter as mset
import regex as re
from nltk.corpus import stopwords
from scipy.spatial.distance import cdist

"""
New vectorizer for usage with stop words removal. 
"""
def vectorize(dim, glove, data, is_train):
    q1_arr = data.question1.values
    q2_arr = data.question2.values

    nbr_samples = len(q1_arr)
    totaltime = 0
    percentage = 0
    nbr_features = 15
    start = time.time()
    vectorized_words = np.zeros((len(q1_arr), dim * 2 + nbr_features))
    for i in range(0, len(q1_arr)):
        q1_vec = np.zeros((1, dim))
        q2_vec = np.zeros((1, dim))
        features = np.zeros((nbr_features))

        # Calculate % complete and estimated time left
        if i % int((nbr_samples / 100)) == 0:
            totaltime += (time.time() - start)
            time_estimate = totaltime / (percentage + 1) * (100 - percentage)
            min = int(time_estimate / 60)
            sec = int(time_estimate - min * 60)
            print("Vectorizing...{0}% complete. Estimated time: {1}:{2}".format(str(percentage), str(min), str(sec), end='\r'))
            start = time.time()
            percentage += 1

        q1 = q1_arr[i].lower()
        q2 = q2_arr[i].lower()

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

        vectorized_words[i, :dim] = q1_vec
        vectorized_words[i, dim:dim*2] = q2_vec
        vectorized_words[i, dim*2:] = features

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
Returns the number of words not in common
"""
def get_common(q1, q2):
    common = list((mset(q1) & mset(q2)).elements())
    return len(common)