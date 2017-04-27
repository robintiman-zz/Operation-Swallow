import regex as re
import numpy as np
import scipy.spatial.distance as sp
import time
import math
import numpy as np
import regex as re

"""
Vectorizing each question of the the csv_array using glove
Return: A vectorized csv_vec matrix where rows are each example and columns are the data the with format:
            [][0]: Correct value
            [][1]: Row number
            [][2]: Question 1 ID
            [][3]: Question 2 ID
            [][4-dim]: N-dimensional vector of vectorized question 1
            [][4+dim-end]: N-dimensional vector of vectorized question 2
"""
def vectorize(csv_array, glove, dim, is_train):
    nbr_basic_features = 8
    nbr_distance_features = 5
    csv_vec = np.zeros((len(csv_array), len(csv_array[0]) + 2*dim - 2 + nbr_basic_features + nbr_distance_features))
    percentage = 0
    start = time.time()
    totaltime = 0
    data_range = len(csv_array)

    if is_train:
        q1_index = 3
        q2_index = 4
    else:
        q1_index = 1
        q2_index = 2

    for i in range(1, data_range):

        # Calculate % complete and estimated time left
        if i%int((data_range/100)) == 0:
            totaltime += (time.time()-start)
            time_estimate = totaltime/(percentage+1)*(100-percentage)
            min = int(time_estimate/60)
            sec = int(time_estimate - min*60)
            print("Vectorizing..." + str(percentage) + "% complete. Estimated time: " + str(min) + ":" + str(sec))
            start = time.time()
            percentage += 1

        # Lower case each question
        q1 = str.lower(csv_array[i][q1_index])
        q2 = str.lower(csv_array[i][q2_index])

        # Regex separating each word of the questions in a vector
        q1_words = re.findall(r'\p{L}+', q1)
        q2_words = re.findall(r'\p{L}+', q2)

        # Initialize numpy vectors
        q1_vec = np.zeros((1, dim))
        q2_vec = np.zeros((1, dim))

        # Add all vectorized words from glove in each question vector. If wrongly spelled, ignore it
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

        # Normalize the vectorized questions
        q1_norm = np.linalg.norm(q1_vec)
        q2_norm = np.linalg.norm(q2_vec)
        if q1_norm > 0 and q2_norm > 0:
            q1_vec = np.divide(q1_vec, q1_norm)
            q2_vec = np.divide(q2_vec, q2_norm)

        # Put together the whole final matrix, with train or test taken in account
        assembleFinalMatrix(csv_vec, csv_array, dim, is_train, i, q1_vec, q2_vec, q1, q2, nbr_basic_features, nbr_distance_features)

    return csv_vec

def assembleFinalMatrix(csv_vec, csv_array, dim, is_train, i, q1_vec, q2_vec, q1, q2, nbr_basic_features, nbr_distance_features):
    #Add initial values
    if is_train:
        csv_vec[i][0] = csv_array[i][5]
        csv_vec[i][1] = csv_array[i][0]
        csv_vec[i][2] = csv_array[i][1]
        csv_vec[i][3] = csv_array[i][2]
        index_offset = 4
    else:
        csv_vec[i][0] = csv_array[i][0]
        index_offset = 1

    #Add vectorized values
    for j in range(0, dim):
        csv_vec[i][j + index_offset] = q1_vec[0][j]
        csv_vec[i][j + index_offset + dim] = q2_vec[0][j]

    #Add basic features
    basic_features = get_basic_features(q1, q2)
    for j in range(0, nbr_basic_features):
        csv_vec[i][j + index_offset + 2*dim] = basic_features[j]

    #Add distance features
    distance_features = get_glove_distance_features(q1_vec, q2_vec)
    for j in range(0, nbr_distance_features):
        csv_vec[i][j + index_offset + 2 * dim + nbr_basic_features] = distance_features[j]

def get_basic_features(q1, q2):
    q1 = str.lower(q1)
    q2 = str.lower(q2)
    q1_words = re.findall(r'\p{L}+', q1)
    q2_words = re.findall(r'\p{L}+', q2)

    q1_length = len(q1)
    q2_length = len(q2)
    length_diff = math.fabs(q1_length - q2_length)
    q1_length_no_space = len(q1.replace(" ", ""))
    q2_length_no_space = len(q2.replace(" ", ""))
    q1_nbr_words = len(q1_words)
    q2_nbr_words = len(q2_words)
    nbr_common_words = len(list(set(q1_words).intersection(q2_words)))
    #print("q1_length: " + str(q1_length)
          #+ "\nq2_length: " + str(q2_length)
          #+ "\nlength_diff: " + str(length_diff)
          #+ "\nq1_length_no_space: " + str(q1_length_no_space)
          #+ "\nq2_length_no_space: " + str(q2_length_no_space)
          #+ "\nq1_nbr_words: " + str(q1_nbr_words)
          #+ "\nq2_nbr_words: " + str(q2_nbr_words)
          #+ "\nnbr_common_words: " + str(nbr_common_words))
    return q1_length, q2_length, length_diff, q1_length_no_space, q2_length_no_space, q1_nbr_words, q2_nbr_words, nbr_common_words

def get_glove_distance_features(q1_vec, q2_vec):
    if not (np.array_equal(q1_vec, np.zeros((1, 50))) or np.array_equal(q2_vec, np.zeros((1, 50)))):
    #word_mover_dist =
    #normalized_word_mover_dist =
        cosine_dist = sp.cosine(q1_vec, q2_vec)
        manhattan_dist = sp.cityblock(q1_vec, q2_vec)
    #jaccard_similarity = sp.spatial.distance.jaccard(q1_vec, q2_vec)
        canberra_dist = sp.canberra(q1_vec, q2_vec)
        minkowski_dist = sp.minkowski(q1_vec, q2_vec, 3)
        braycurtis_dist = sp.braycurtis(q1_vec, q2_vec)
    else:
        cosine_dist = 0
        manhattan_dist = 0
        canberra_dist = 0
        minkowski_dist = 0
        braycurtis_dist = 0

    #q1_skew = sp.stats.skew(q1_vec)
    #q2_skew = sp.stats.skew(q2_vec)
    #q1_kurtosis = sp.stats.kurtosis(q1_vec)
    #q2_kurtosis = sp.stats.kurtosis(q2_vec)

    #print("cosine_dist: " + str(cosine_dist)
          #+ "\nmanhattan_dist: " + str(manhattan_dist)
          #+ #"\njaccard_similarity: " + str(jaccard_similarity)
          #+ "\ncanberra_dist: " + str(canberra_dist)
          #+ "\nminkowski_dist: " + str(minkowski_dist)
          #+ "\nbraycurtis_dist: " + str(braycurtis_dist)
          #+ "\nq1_skew: " + str(q1_skew)
          #+ "\nq2_skew: " + str(q2_skew)
          #+ "\nq1_kurtosis: " + str(q1_kurtosis)
          #+ "\nq2_kurtosis: " + str(q2_kurtosis)
          #)
    return cosine_dist, manhattan_dist, canberra_dist, minkowski_dist, braycurtis_dist