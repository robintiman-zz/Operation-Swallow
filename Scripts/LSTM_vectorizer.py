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
    start = time.time()
    #Skip samples above max_n_words due to too much memory required
    max_n_words = 70
    nbr_cuts = 0
    split = 1
    if not(is_train): split = 4

    for n in range(0, split):
        vectorized_words = None
        vectorized_words = np.zeros((nbr_samples/split, max_n_words, dim*2))

        for i in range(int((nbr_samples/split)*n), int((nbr_samples/split)*(n+1))):
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

            # Remove stopwords
            q1_words, q2_words = remove_stop(q1_words, q2_words)

            #Cut a question if it's too long
            if len(q1_words) > max_n_words or len(q2_words) > max_n_words:
                nbr_cuts += 1
            if len(q1_words) > max_n_words:
                nbr_cuts += 1
                q1_words_cut = []
                for j in range(0, max_n_words):
                    q1_words_cut.append(q1_words[j])
                q1_words = q1_words_cut
            if len(q2_words) > max_n_words:
                q2_words_cut = []
                for j in range(0, max_n_words):
                    q2_words_cut.append(q2_words[j])
                q2_words = q2_words_cut

            # GloVe features
            for j in range(0, len(q1_words)):
                try:
                    vectorized_words[i - int((nbr_samples / split) * n), j, :dim] = glove[q1_words[j]]
                except KeyError:
                    continue
            for j in range(0, len(q2_words)):
                try:
                    vectorized_words[i - int((nbr_samples / split) * n), j, dim:dim*2] = glove[q2_words[j]]
                except KeyError:
                    continue

        #Write to file

        if is_train:
            print("Writing to file...")
            np.save("/media/calle/SSD/OperationSwallowData/LSTM_train_vector", vectorized_words)
        else:
            print("Writing batch", n+1, "to file...")
            np.save("/media/calle/7E549DAA549D6625/OperationSwallowData/LSTM_test_vector_" + str(n+1), vectorized_words)
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